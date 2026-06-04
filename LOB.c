#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum {
    ORDER_MATCHING_ALGORITHM_PRICE_TIME_PRIORITY,
    ORDER_MATCHING_ALGORITHM_PRO_RATA
} OrderMatchingAlgorithm;

typedef struct {
    int64_t price;
    size_t order_index;
    bool is_bid;
} OrderReference;

typedef struct {
    uint64_t quantity;
    size_t previous_order_index;
    size_t next_order_index;
    OrderReference* reference;
} Order;

typedef struct {
    uint64_t total_quantity;
    size_t orders_head;
    size_t orders_tail;
    size_t capacity;
    Order* orders;
    uint32_t total_orders;
} PriceBucket;

typedef struct {
    int64_t best_price;
    int64_t worst_price;
    size_t best_price_bucket_index;
    size_t capacity;
    PriceBucket* price_buckets;
} Side;

typedef struct {
    OrderMatchingAlgorithm order_matching_algorithm;
    int64_t tick_size;
    Side bids;
    Side asks;
} LimitOrderBook;

Order construct_order() {
    Order order;
    order.quantity = 0;
    order.previous_order_index = SIZE_MAX;
    order.next_order_index = SIZE_MAX;
    order.reference = NULL;
    return order;
}

PriceBucket construct_price_bucket() {
    PriceBucket price_bucket;
    price_bucket.total_quantity = 0;
    price_bucket.orders_head = SIZE_MAX;
    price_bucket.orders_tail = SIZE_MAX;
    price_bucket.capacity = 0;
    price_bucket.orders = NULL;
    price_bucket.total_orders = 0;
    return price_bucket;
}

Side construct_side() {
    Side side;
    side.best_price = INT64_MAX;
    side.worst_price = INT64_MAX;
    side.best_price_bucket_index = SIZE_MAX;
    side.capacity = 0;
    side.price_buckets = NULL;
    return side;
}

LimitOrderBook construct_limit_order_book(int64_t tick_size) {
    LimitOrderBook lob;
    if(tick_size <= 0) { tick_size = 1; }
    else { lob.tick_size = tick_size; }
    lob.bids = construct_side();
    lob.asks = construct_side();
    return lob;
}

void destroy_order(Order* order) {
    free(order->reference);
    order->reference = NULL;

    order->quantity = 0;
    order->previous_order_index = SIZE_MAX;
    order->next_order_index = SIZE_MAX;
}

void destroy_price_bucket(PriceBucket* price_bucket) {
    for(size_t i = 0; i < price_bucket->capacity; i++) { destroy_order(&price_bucket->orders[i]); }
    free(price_bucket->orders);
    price_bucket->orders = NULL;

    price_bucket->total_quantity = 0;
    price_bucket->orders_head = SIZE_MAX;
    price_bucket->orders_tail = SIZE_MAX;
    price_bucket->capacity = 0;
    price_bucket->total_orders = 0;
}

void destroy_side(Side* side) {
    for(size_t i = 0; i < side->capacity; i++) { destroy_price_bucket(&side->price_buckets[i]); }
    free(side->price_buckets);
    side->price_buckets = NULL;

    side->best_price = INT64_MAX;
    side->worst_price = INT64_MAX;
    side->best_price_bucket_index = SIZE_MAX;
    side->capacity = 0;
}

void destroy_limit_order_book(LimitOrderBook* lob) {
    destroy_side(&lob->bids);
    destroy_side(&lob->asks);
    lob->tick_size = 0;
}

void rebuild_orders(PriceBucket* price_bucket, size_t resize_increment) {
    if(resize_increment < 1) {
        resize_increment = 1;
    }

    size_t new_capacity = price_bucket->capacity;
    if(new_capacity <= (((size_t)price_bucket->total_orders) + resize_increment)) {
        new_capacity += resize_increment;
    }
    Order* new_orders = malloc(new_capacity * sizeof(Order));

    size_t old_index = price_bucket->orders_head;
    size_t new_index = 0;
    for(uint32_t i = 0; i < price_bucket->total_orders; i++) {
        new_orders[new_index].quantity = price_bucket->orders[old_index].quantity;
        new_orders[new_index].previous_order_index = (i > 0) ? (i - 1) : SIZE_MAX;
        new_orders[new_index].next_order_index = ((i+1) < price_bucket->total_orders) ? (i + 1) : SIZE_MAX;
        new_orders[new_index].reference = price_bucket->orders[old_index].reference;
        new_orders[new_index].reference->order_index = new_index;

        old_index = price_bucket->orders[old_index].next_order_index;
        new_index++;
    }

    for(size_t i = new_index; i < new_capacity; i++) {
        new_orders[i] = construct_order();
    }

    price_bucket->orders_head = (price_bucket->total_orders > 0) ? 0 : SIZE_MAX;
    price_bucket->orders_tail = (price_bucket->total_orders > 0) ? (new_index - 1) : SIZE_MAX;
    price_bucket->capacity = new_capacity;
    free(price_bucket->orders);
    price_bucket->orders = new_orders;
}

void rebuild_price_buckets(Side* side, size_t resize_increment) {
    size_t new_capacity = side->capacity + resize_increment;
    size_t new_best_price_bucket_index = new_capacity / 2;
    PriceBucket* new_price_buckets = malloc(new_capacity * sizeof(PriceBucket));

    size_t new_index = new_best_price_bucket_index;
    size_t old_index = side->best_price_bucket_index;
    for(size_t i = 0; i < side->capacity; i++) {
        if(old_index >= side->capacity) { old_index -= side->capacity; }
        if(new_index >= new_capacity) { new_index -= new_capacity; }
        new_price_buckets[new_index] = side->price_buckets[old_index];
        new_index++;
        old_index++;
    }

    for(size_t i = side->capacity; i < new_capacity; i++) {
        if(new_index >= new_capacity) { new_index -= new_capacity; }
        new_price_buckets[new_index] = construct_price_bucket();
        new_index++;
    }

    side->best_price_bucket_index = new_best_price_bucket_index;
    side->capacity = new_capacity;
    free(side->price_buckets);
    side->price_buckets = new_price_buckets;
}

size_t get_price_bucket_index(Side* side, bool is_bid, int64_t price, int64_t tick_size) {
    if(side->best_price_bucket_index == SIZE_MAX) {
        if(side->capacity == 0) {
            rebuild_price_buckets(side, 100);
        }
        else { side->best_price_bucket_index = side->capacity / 2; }
        side->best_price = price;
        side->worst_price = price;
        return side->best_price_bucket_index;
    }

    int64_t direction = is_bid ? 1 : -1;

    if(direction * (side->best_price - price) >= 0) {
        size_t offset = (size_t)(direction * (side->best_price - price) / tick_size);
        if((offset + 1) > side->capacity) {
            size_t extra_capacity_needed = (offset + 1) - side->capacity;
            size_t resize_increment = (extra_capacity_needed > 100) ? extra_capacity_needed : 100;
            rebuild_price_buckets(side, resize_increment);
        }
        if(direction * (side->worst_price - price) > 0) { side->worst_price = price; }

        size_t price_bucket_index = side->best_price_bucket_index + offset;
        if(price_bucket_index >= side->capacity) { price_bucket_index -= side->capacity; }
        return price_bucket_index;
    }
    else {
        size_t offset = (size_t)(direction * (price - side->best_price) / tick_size);
        size_t worst_price_offset = (size_t)(direction * (side->best_price - side->worst_price) / tick_size);
        size_t total_capacity_needed = worst_price_offset + offset + 1;
        if(total_capacity_needed > side->capacity) {
            size_t extra_capacity_needed = total_capacity_needed - side->capacity;
            size_t resize_increment = (extra_capacity_needed > 100) ? extra_capacity_needed : 100;
            rebuild_price_buckets(side, resize_increment);
        }

        size_t new_best_index;
        if(offset <= side->best_price_bucket_index) { new_best_index = side->best_price_bucket_index - offset; }
        else { new_best_index = side->capacity - (offset - side->best_price_bucket_index); }
        side->best_price_bucket_index = new_best_index;
        side->best_price = price;
        return new_best_index;
    }
}

OrderReference* add_order_to_limit_order_book(LimitOrderBook* lob, bool is_bid, int64_t price, uint64_t quantity) {
    Side* side = is_bid ? &lob->bids : &lob->asks;
    size_t price_bucket_index = get_price_bucket_index(side, is_bid, price, lob->tick_size);
    PriceBucket* bucket = &side->price_buckets[price_bucket_index];

    size_t old_tail;
    size_t new_tail;
    if(bucket->total_orders == 0) {
        if(bucket->orders == NULL) {
            rebuild_orders(bucket, 1000);
        }
        old_tail = SIZE_MAX;
        new_tail = 0;
        bucket->orders_head = 0;
    }
    else {
        old_tail = bucket->orders_tail;
        new_tail = old_tail + 1;
        if(new_tail >= bucket->capacity) {
            new_tail -= bucket->capacity;
        }
        if(new_tail == bucket->orders_head) {
            rebuild_orders(bucket, 1000);
            old_tail = bucket->orders_tail;
            new_tail = old_tail + 1;
        }
        bucket->orders[old_tail].next_order_index = new_tail;
    }

    Order* order = &bucket->orders[new_tail];
    order->quantity = quantity;
    order->previous_order_index = old_tail;
    order->next_order_index = SIZE_MAX;
    order->reference = malloc(sizeof(OrderReference));
    order->reference->price = price;
    order->reference->order_index = new_tail;
    order->reference->is_bid = is_bid;

    bucket->total_quantity += quantity;
    bucket->orders_tail = new_tail;
    bucket->total_orders++;

    return order->reference;
}

void cancel_order(Side* side, bool is_bid, int64_t price, size_t order_index, int64_t tick_size) {
    int64_t direction = is_bid ? 1 : -1;

    size_t price_bucket_index = get_price_bucket_index(side, is_bid, price, tick_size);
    PriceBucket* price_bucket = &side->price_buckets[price_bucket_index];
    Order* order = &price_bucket->orders[order_index];

    uint64_t quantity = side->price_buckets[price_bucket_index].orders[order_index].quantity;
    size_t previous_order_index = order->previous_order_index;
    size_t next_order_index = order->next_order_index;

    if(previous_order_index != SIZE_MAX) {
        price_bucket->orders[previous_order_index].next_order_index = next_order_index;
    }
    else {
        price_bucket->orders_head = next_order_index;
    }
    if(next_order_index != SIZE_MAX) {
        price_bucket->orders[next_order_index].previous_order_index = previous_order_index;
    }
    else {
        price_bucket->orders_tail = previous_order_index;
    }

    destroy_order(order);
    price_bucket->total_quantity -= quantity;
    price_bucket->total_orders--;

    if(price_bucket->total_orders == 0) {
        size_t worst_offset = (size_t)(direction * (side->best_price - side->worst_price) / tick_size);

        if(price == side->best_price) {
            size_t tick_offset;
            for (tick_offset = 1; tick_offset <= worst_offset; tick_offset++) {
                size_t index = side->best_price_bucket_index + tick_offset;
                if (index >= side->capacity) { index -= side->capacity; }
                if (side->price_buckets[index].total_orders > 0) {
                    side->best_price_bucket_index = index;
                    side->best_price -= direction * ((int64_t)tick_offset) * tick_size;
                    break;
                }
            }
            if (tick_offset > worst_offset) {
                side->best_price = INT64_MAX;
                side->worst_price = INT64_MAX;
                side->best_price_bucket_index = SIZE_MAX;
            }
        }
        else if (price == side->worst_price) {
            for (size_t offset = worst_offset; offset > 0; offset--) {
                size_t index = side->best_price_bucket_index + (offset - 1);
                if (index >= side->capacity) { index -= side->capacity; }
                if (side->price_buckets[index].total_orders > 0) {
                    side->worst_price = side->best_price - direction * ((int64_t)(offset - 1)) * tick_size;
                    break;
                }
            }
        }
    }
}

uint64_t match_market_order_using_time_price_priority(Side* side, bool is_bid, int64_t tick_size, int64_t limit_price, uint64_t quantity, bool is_fill_or_kill) {
    int64_t direction = is_bid ? 1 : -1;
    uint64_t unfilled_quantity = quantity;

    // Empty book → nothing to match
    if(side->best_price_bucket_index == SIZE_MAX) { return unfilled_quantity; }
    // limit_price doesn't cross best_price → nothing to match
    if(direction * (side->best_price - limit_price) < 0) { return unfilled_quantity; }

    size_t worst_price_offset = (size_t)(direction * (side->best_price - side->worst_price) / tick_size);
    size_t limit_price_offset = (size_t)(direction * (side->best_price - limit_price) / tick_size);
    if(limit_price_offset > worst_price_offset) { limit_price_offset = worst_price_offset; }

    if(is_fill_or_kill) {
        uint64_t available_quantity = 0;
        for(size_t i = 0; i <= limit_price_offset; i++) {
            size_t price_bucket_index = side->best_price_bucket_index + i;
            if(price_bucket_index >= side->capacity) { price_bucket_index -= side->capacity; }
            available_quantity += side->price_buckets[price_bucket_index].total_quantity;
        }
        if(available_quantity < unfilled_quantity) { return unfilled_quantity; }
    }

    for(size_t price_bucket = 0; price_bucket < side->capacity; price_bucket++) {
        if(unfilled_quantity == 0) { break; }
        if(side->best_price_bucket_index == SIZE_MAX) { break; }
        if(direction * (side->best_price - limit_price) < 0) { break; }

        size_t best_price_bucket_index = side->best_price_bucket_index;
        PriceBucket* bucket = &side->price_buckets[best_price_bucket_index];
        uint32_t total_orders_at_start = bucket->total_orders;

        for(uint32_t order = 0; order < total_orders_at_start; order++) {
            if(unfilled_quantity == 0) { break; }
            size_t order_index = bucket->orders_head;
            uint64_t order_quantity = bucket->orders[order_index].quantity;
            uint64_t fill_quantity = (unfilled_quantity < order_quantity) ? unfilled_quantity : order_quantity;

            unfilled_quantity -= fill_quantity;
            bucket->orders[order_index].quantity -= fill_quantity;
            bucket->total_quantity -= fill_quantity;

            if(bucket->orders[order_index].quantity == 0) {
                size_t next_order_index = bucket->orders[order_index].next_order_index;
                if(next_order_index != SIZE_MAX) {
                    bucket->orders[next_order_index].previous_order_index = SIZE_MAX;
                }
                else {
                    bucket->orders_tail = SIZE_MAX;
                }
                bucket->orders_head = next_order_index;
                destroy_order(&bucket->orders[order_index]);
                bucket->total_orders--;
            }
        }

        if(bucket->total_orders == 0) {
            size_t worst_offset = (size_t)(direction * (side->best_price - side->worst_price) / tick_size);
            size_t tick_offset;
            for(tick_offset = 1; tick_offset <= worst_offset; tick_offset++) {
                size_t index = side->best_price_bucket_index + tick_offset;
                if(index >= side->capacity) { index -= side->capacity; }
                if(side->price_buckets[index].total_orders > 0) {
                    side->best_price_bucket_index = index;
                    side->best_price -= direction * ((int64_t)tick_offset) * tick_size;
                    break;
                }
            }
            if(tick_offset > worst_offset) {
                side->best_price = INT64_MAX;
                side->worst_price = INT64_MAX;
                side->best_price_bucket_index = SIZE_MAX;
            }
        }
    }

    return unfilled_quantity;
}

uint64_t match_market_order_pro_rata(Side* side, bool is_bid, int64_t tick_size, int64_t limit_price, uint64_t quantity, bool is_fill_or_kill, uint64_t minimum_fill_quantity) {
    int64_t direction = is_bid ? 1 : -1;
    uint64_t unfilled_quantity = quantity;

    if(side->best_price_bucket_index == SIZE_MAX) { return unfilled_quantity; }
    if(direction * (side->best_price - limit_price) < 0) { return unfilled_quantity; }

    size_t worst_price_offset = (size_t)(direction * (side->best_price - side->worst_price) / tick_size);
    size_t limit_price_offset = (size_t)(direction * (side->best_price - limit_price) / tick_size);
    if(limit_price_offset > worst_price_offset) { limit_price_offset = worst_price_offset; }

    if(is_fill_or_kill) {
        uint64_t available_quantity = 0;
        for(size_t i = 0; i <= limit_price_offset; i++) {
            size_t price_bucket_index = side->best_price_bucket_index + i;
            if(price_bucket_index >= side->capacity) { price_bucket_index -= side->capacity; }
            available_quantity += side->price_buckets[price_bucket_index].total_quantity;
        }
        if(available_quantity < unfilled_quantity) { return unfilled_quantity; }
    }

    for(size_t price_bucket = 0; price_bucket < side->capacity; price_bucket++) {
        if(unfilled_quantity == 0) { break; }
        if(side->best_price_bucket_index == SIZE_MAX) { break; }
        if(direction * (side->best_price - limit_price) < 0) { break; }

        size_t best_price_bucket_index = side->best_price_bucket_index;
        PriceBucket* bucket = &side->price_buckets[best_price_bucket_index];
        uint64_t total_quantity_in_bucket = bucket->total_quantity;

        if(total_quantity_in_bucket <= unfilled_quantity) {
            unfilled_quantity -= total_quantity_in_bucket;
            bucket->total_quantity = 0;

            size_t order_index = bucket->orders_head;
            uint32_t orders_to_destroy = bucket->total_orders;
            for(uint32_t order_num = 0; order_num < orders_to_destroy; order_num++) {
                size_t next_order_index = bucket->orders[order_index].next_order_index;
                destroy_order(&bucket->orders[order_index]);
                order_index = next_order_index;
            }
            bucket->orders_head = SIZE_MAX;
            bucket->orders_tail = SIZE_MAX;
            bucket->total_orders = 0;
        }
        else {
            uint64_t starting_unfilled_quantity = unfilled_quantity;
            uint64_t starting_bucket_quantity = total_quantity_in_bucket;
            uint32_t total_orders_at_start = bucket->total_orders;
            size_t order_index = bucket->orders_head;

            for(uint32_t order_num = 0; order_num < total_orders_at_start; order_num++) {
                size_t previous_order_index = bucket->orders[order_index].previous_order_index;
                size_t next_order_index = bucket->orders[order_index].next_order_index;
                uint64_t order_quantity = bucket->orders[order_index].quantity;

                uint64_t fill_quantity = 0;
                uint64_t running_remainder = 0;
                for(size_t bit = 64; bit-- > 0; ) {
                    fill_quantity += fill_quantity;
                    running_remainder += running_remainder;
                    if(running_remainder >= starting_bucket_quantity) {
                        running_remainder -= starting_bucket_quantity;
                        fill_quantity += 1;
                    }
                    if((order_quantity >> bit) & 1) {
                        running_remainder += starting_unfilled_quantity;
                        if(running_remainder >= starting_bucket_quantity) {
                            running_remainder -= starting_bucket_quantity;
                            fill_quantity += 1;
                        }
                    }
                }
                if(fill_quantity < minimum_fill_quantity) { fill_quantity = 0; }
                if(fill_quantity > order_quantity) { fill_quantity = order_quantity; }

                if(fill_quantity > 0) {
                    bucket->orders[order_index].quantity -= fill_quantity;
                    bucket->total_quantity -= fill_quantity;
                    unfilled_quantity -= fill_quantity;

                    if(bucket->orders[order_index].quantity == 0) {
                        if(previous_order_index != SIZE_MAX) {
                            bucket->orders[previous_order_index].next_order_index = next_order_index;
                        }
                        else {
                            bucket->orders_head = next_order_index;
                        }
                        if(next_order_index != SIZE_MAX) {
                            bucket->orders[next_order_index].previous_order_index = previous_order_index;
                        }
                        else {
                            bucket->orders_tail = previous_order_index;
                        }
                        destroy_order(&bucket->orders[order_index]);
                        bucket->total_orders--;
                    }
                }
                order_index = next_order_index;
            }

            uint32_t orders_remaining = bucket->total_orders;
            for(uint32_t order_num = 0; order_num < orders_remaining; order_num++) {
                if(unfilled_quantity == 0) { break; }
                size_t leftover_order_index = bucket->orders_head;
                uint64_t order_quantity = bucket->orders[leftover_order_index].quantity;
                uint64_t fill_quantity = (unfilled_quantity < order_quantity) ? unfilled_quantity : order_quantity;

                unfilled_quantity -= fill_quantity;
                bucket->orders[leftover_order_index].quantity -= fill_quantity;
                bucket->total_quantity -= fill_quantity;

                if(bucket->orders[leftover_order_index].quantity == 0) {
                    size_t next_order_index = bucket->orders[leftover_order_index].next_order_index;
                    if(next_order_index != SIZE_MAX) {
                        bucket->orders[next_order_index].previous_order_index = SIZE_MAX;
                    }
                    else {
                        bucket->orders_tail = SIZE_MAX;
                    }
                    bucket->orders_head = next_order_index;
                    destroy_order(&bucket->orders[leftover_order_index]);
                    bucket->total_orders--;
                }
            }
        }

        if(bucket->total_orders == 0) {
            size_t worst_offset = (size_t)(direction * (side->best_price - side->worst_price) / tick_size);
            size_t tick_offset;
            for(tick_offset = 1; tick_offset <= worst_offset; tick_offset++) {
                size_t index = side->best_price_bucket_index + tick_offset;
                if(index >= side->capacity) { index -= side->capacity; }
                if(side->price_buckets[index].total_orders > 0) {
                    side->best_price_bucket_index = index;
                    side->best_price -= direction * ((int64_t)tick_offset) * tick_size;
                    break;
                }
            }
            if(tick_offset > worst_offset) {
                side->best_price = INT64_MAX;
                side->worst_price = INT64_MAX;
                side->best_price_bucket_index = SIZE_MAX;
            }
        }
    }

    return unfilled_quantity;
}

OrderReference* execute_order(LimitOrderBook* lob, bool is_bid, int64_t price, uint64_t quantity, bool is_fill_or_kill, uint64_t minimum_fill_quantity) {
    Side* opposite_side = is_bid ? &lob->asks : &lob->bids;
    bool opposite_is_bid = !is_bid;

    uint64_t unfilled_quantity;
    if(lob->order_matching_algorithm == ORDER_MATCHING_ALGORITHM_PRICE_TIME_PRIORITY) {
        unfilled_quantity = match_market_order_using_time_price_priority(opposite_side, opposite_is_bid, lob->tick_size, price, quantity, is_fill_or_kill);
    }
    if(lob->order_matching_algorithm == ORDER_MATCHING_ALGORITHM_PRO_RATA) {
        unfilled_quantity = match_market_order_pro_rata(opposite_side, opposite_is_bid, lob->tick_size, price, quantity, is_fill_or_kill, minimum_fill_quantity);
    }

    if(unfilled_quantity == 0) { return NULL; }
    if(is_fill_or_kill) { return NULL; }

    return add_order_to_limit_order_book(lob, is_bid, price, unfilled_quantity);
}
