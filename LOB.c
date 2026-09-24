#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define LIMIT_ORDER_BOOK_BID ((int8_t)1)
#define LIMIT_ORDER_BOOK_ASK ((int8_t)-1)

typedef struct {
    int8_t price_direction;
    int64_t price;
    size_t order_index;
} LimitOrderBook_OrderReference;

typedef struct {
    uint64_t total_orders;
    uint64_t total_quantity;

    size_t orders_head_index;
    size_t orders_tail_index;
    size_t orders_capacity;

    size_t order_references_head_index;
    size_t order_references_tail_index;
    size_t order_references_capacity;

    uint64_t* orders;
    LimitOrderBook_OrderReference* order_references;
} LimitOrderBook_PriceBucket;

typedef struct {
    int64_t best_price;
    int64_t worst_price;

    size_t best_price_bucket_index;
    size_t price_buckets_capacity;

    LimitOrderBook_PriceBucket* price_buckets;
} LimitOrderBook_Side;

typedef struct {
    int64_t total_money;

    size_t total_order_references;
    size_t order_references_head_index;
    size_t order_references_tail_index;
    size_t order_references_capacity;

    LimitOrderBook_OrderReference** order_references;
} LimitOrderBook_TraderAccount;

typedef struct {
    int64_t tick_size;
    size_t total_trader_accounts;

    LimitOrderBook_Side asks;
    LimitOrderBook_Side bids;

    LimitOrderBook_TraderAccount* trader_accounts;
} LimitOrderBook;

LimitOrderBook_OrderReference create_limit_order_book_order_reference() {
    LimitOrderBook_OrderReference order_reference;
    order_reference.price_direction = 0;
    order_reference.price = UINT64_MAX;
    order_reference.order_index = SIZE_MAX;

    return order_reference;
}

void destroy_limit_order_book_order_reference(LimitOrderBook_OrderReference* order_reference) {
    order_reference->price_direction = 0;
    order_reference->price = UINT64_MAX;
    order_reference->order_index = SIZE_MAX;
}

LimitOrderBook_PriceBucket create_limit_order_book_price_bucket() {
    LimitOrderBook_PriceBucket price_bucket;
    price_bucket.total_orders = 0;
    price_bucket.total_quantity = 0;

    price_bucket.orders_head_index = SIZE_MAX;
    price_bucket.orders_tail_index = SIZE_MAX;
    price_bucket.orders_capacity = SIZE_MAX;

    price_bucket.order_references_head_index = SIZE_MAX;
    price_bucket.order_references_tail_index = SIZE_MAX;
    price_bucket.order_references_capacity = SIZE_MAX;

    price_bucket.orders = NULL;
    price_bucket.order_references = NULL;

    return price_bucket;
}

void destroy_limit_order_book_price_bucket(LimitOrderBook_PriceBucket* price_bucket) {
    free(price_bucket->orders);
    free(price_bucket->order_references);
    price_bucket->orders = NULL;
    price_bucket->order_references = NULL;

    price_bucket->total_orders = 0;
    price_bucket->total_quantity = 0;

    price_bucket->orders_head_index = SIZE_MAX;
    price_bucket->orders_tail_index = SIZE_MAX;
    price_bucket->orders_capacity = SIZE_MAX;

    price_bucket->order_references_head_index = SIZE_MAX;
    price_bucket->order_references_tail_index = SIZE_MAX;
    price_bucket->order_references_capacity = SIZE_MAX;
}

LimitOrderBook_Side create_limit_order_book_side() {
    LimitOrderBook_Side side;
    side.best_price = INT64_MAX;
    side.worst_price = INT64_MAX;

    side.best_price_bucket_index = SIZE_MAX;
    side.price_buckets_capacity = 0;

    side.price_buckets = NULL;

    return side;
}

void destroy_limit_order_book_side(LimitOrderBook_Side* side) {
    for(size_t i = 0; i < side->price_buckets_capacity; i++) {
        destroy_limit_order_book_price_bucket(&side->price_buckets[i]);
    }
    free(side->price_buckets);
    side->price_buckets = NULL;

    side->best_price = INT64_MAX;
    side->worst_price = INT64_MAX;

    side->best_price_bucket_index = SIZE_MAX;
    side->price_buckets_capacity = 0;
}

LimitOrderBook_TraderAccount create_limit_order_book_trader_account() {
    LimitOrderBook_TraderAccount trader_account;
    trader_account.total_money = 0;

    trader_account.total_order_references = 0;
    trader_account.order_references_head_index = SIZE_MAX;
    trader_account.order_references_tail_index = SIZE_MAX;
    trader_account.order_references_capacity = 0;

    trader_account.order_references = NULL;

    return trader_account;
}

void destroy_limit_order_book_trader_account(LimitOrderBook_TraderAccount* trader_account) {
    for(size_t i = 0; i < trader_account->order_references_capacity; i++) {
        trader_account->order_references[i] = NULL;
    }
    trader_account->order_references = NULL;

    trader_account->total_money = 0;

    trader_account->total_order_references = 0;
    trader_account->order_references_head_index = SIZE_MAX;
    trader_account->order_references_tail_index = SIZE_MAX;
    trader_account->order_references_capacity = 0;
}

LimitOrderBook create_limit_order_book() {
    LimitOrderBook limit_order_book;
    limit_order_book.tick_size = 1;
    limit_order_book.total_trader_accounts = 0;

    limit_order_book.asks = create_limit_order_book_side();
    limit_order_book.bids = create_limit_order_book_side();

    limit_order_book.trader_accounts = NULL;

    return limit_order_book;
}

void destroy_limit_order_book(LimitOrderBook* limit_order_book) {
    destroy_limit_order_book_side(&limit_order_book->asks);
    destroy_limit_order_book_side(&limit_order_book->bids);

    for(size_t i = 0; i < limit_order_book->total_trader_accounts; i++) {
        destroy_limit_order_book_trader_account(&limit_order_book->trader_accounts[i]);
    }
    limit_order_book->trader_accounts = NULL;

    limit_order_book->tick_size = 1;
    limit_order_book->total_trader_accounts = 0;
}

void compact_orders_in_limit_order_book_price_bucket(LimitOrderBook_PriceBucket* price_bucket) {
    size_t new_index = price_bucket->orders_head_index;
    size_t old_index = price_bucket->orders_head_index;
    for(size_t i = 0; i < price_bucket->orders_capacity; i++) {
        if(price_bucket->total_orders == 0) {
            break;
        }

        if(price_bucket->orders[old_index] > 0) {
            if(new_index != old_index) {
                price_bucket->orders[new_index] = price_bucket->orders[old_index];
                price_bucket->orders[old_index] = 0;

                for(size_t j = 0; j < price_bucket->order_references_capacity; j++) {
                    if(price_bucket->order_references[j].order_index == old_index) {
                        price_bucket->order_references[j].order_index = new_index;
                        break;
                    }
                }
            }

            new_index++;
            if(new_index >= price_bucket->orders_capacity) {
                new_index -= price_bucket->orders_capacity;
            }
        }

        old_index++;
        if(old_index >= price_bucket->orders_capacity) {
            old_index -= price_bucket->orders_capacity;
        }
    }

    if(price_bucket->total_orders > 0) {
        if(new_index == 0) {
            price_bucket->orders_tail_index = price_bucket->orders_capacity - 1;
        }
        else {
            price_bucket->orders_tail_index = new_index - 1;
        }
    }
}

void rebuild_limit_order_book_side(LimitOrderBook_Side* side, size_t new_price_buckets_capacity) {
    size_t new_best_price_bucket_index = new_price_buckets_capacity / 2;
    LimitOrderBook_PriceBucket* new_price_buckets = malloc(new_price_buckets_capacity * sizeof(LimitOrderBook_PriceBucket));

    size_t old_price_bucket_index = side->best_price_bucket_index;
    size_t new_price_bucket_index = new_best_price_bucket_index;
    for(size_t i = 0; i < side->price_buckets_capacity; i++) {
        if(i < new_price_buckets_capacity) {
            new_price_buckets[new_price_bucket_index] = side->price_buckets[old_price_bucket_index];
            new_price_bucket_index++;
            if(new_price_bucket_index >= new_price_buckets_capacity) {
                new_price_bucket_index -= new_price_buckets_capacity;
            }
        }
        else {
            destroy_limit_order_book_price_bucket(&side->price_buckets[old_price_bucket_index]);
        }

        old_price_bucket_index++;
        if(old_price_bucket_index >= side->price_buckets_capacity) {
            old_price_bucket_index -= side->price_buckets_capacity;
        }
    }

    for(size_t i = side->price_buckets_capacity; i < new_price_buckets_capacity; i++) {
        new_price_buckets[new_price_bucket_index] = create_limit_order_book_price_bucket();
        new_price_bucket_index++;
        if(new_price_bucket_index >= new_price_buckets_capacity) {
            new_price_bucket_index -= new_price_buckets_capacity;
        }
    }

    free(side->price_buckets);
    side->best_price_bucket_index = new_best_price_bucket_index;
    side->price_buckets_capacity = new_price_buckets_capacity;
    side->price_buckets = new_price_buckets;
}

size_t get_limit_order_book_price_bucket_index(LimitOrderBook_Side* side, int64_t tick_size, int8_t price_direction, int64_t price) {
    const size_t default_resize_increment = 10;
    if(side->best_price == INT64_MAX) {
        if(side->price_buckets_capacity == 0) {
            rebuild_limit_order_book_side(side, default_resize_increment);
        }
        else {
            side->best_price_bucket_index = side->price_buckets_capacity / 2;
        }
        side->best_price = price;
        side->worst_price = price;
        return side->best_price_bucket_index;
    }

    int64_t direction = price_direction;

    if((direction * (side->best_price - price)) >= 0) {
        size_t offset = (size_t)((direction * (side->best_price - price)) / tick_size);
        if((offset + 1) > side->price_buckets_capacity) {
            size_t minimum_resize_increment = offset - side->price_buckets_capacity + 1;
            size_t resize_increment = (minimum_resize_increment > default_resize_increment) ? minimum_resize_increment : default_resize_increment;
            rebuild_limit_order_book_side(side, side->price_buckets_capacity + resize_increment);
        }

        if((direction * (side->worst_price - price)) > 0) {
            side->worst_price = price;
        }

        size_t price_bucket_index = side->best_price_bucket_index + offset;
        if(price_bucket_index >= side->price_buckets_capacity) {
            price_bucket_index -= side->price_buckets_capacity;
        }
        return price_bucket_index;
    }
    else {
        size_t offset = (size_t)((direction * (price - side->best_price)) / tick_size);
        size_t worst_price_offset = (size_t)((direction * (side->best_price - side->worst_price)) / tick_size);
        size_t price_bucket_capacity_needed = worst_price_offset + offset + 1;
        if(price_bucket_capacity_needed > side->price_buckets_capacity) {
            size_t minimum_resize_increment = price_bucket_capacity_needed - side->price_buckets_capacity;
            size_t resize_increment = (minimum_resize_increment > default_resize_increment) ? minimum_resize_increment : default_resize_increment;
            rebuild_limit_order_book_side(side, side->price_buckets_capacity + resize_increment);
        }

        size_t new_best_price_bucket_index;
        if(offset <= side->best_price_bucket_index) {
            new_best_price_bucket_index = side->best_price_bucket_index - offset;
        }
        else {
            new_best_price_bucket_index = side->price_buckets_capacity - (offset - side->best_price_bucket_index);
        }
        side->best_price_bucket_index = new_best_price_bucket_index;
        side->best_price = price;
        return new_best_price_bucket_index;
    }
}

uint64_t get_total_quantity_at_price_in_limit_order_book(LimitOrderBook* limit_order_book, int8_t price_direction, int64_t price) {
    LimitOrderBook_Side* side = (price_direction == LIMIT_ORDER_BOOK_BID) ? &limit_order_book->bids : &limit_order_book->asks;
    int64_t direction = price_direction;

    if(side->best_price_bucket_index == SIZE_MAX) {
        return 0;
    }
    if((direction * (side->best_price - price)) < 0) {
        return 0;
    }
    if((direction * (price - side->worst_price)) < 0) {
        return 0;
    }

    size_t price_bucket_index = get_limit_order_book_price_bucket_index(side, limit_order_book->tick_size, price_direction, price);
    return side->price_buckets[price_bucket_index].total_quantity;
}

uint64_t match_limit_order_book_order_using_time_price_priority(LimitOrderBook* limit_order_book, int8_t price_direction, int64_t limit_price, uint64_t quantity) {
    LimitOrderBook_Side* side = (price_direction == LIMIT_ORDER_BOOK_BID) ? &limit_order_book->bids : &limit_order_book->asks;
    int64_t tick_size = limit_order_book->tick_size;
    int64_t direction = price_direction;

    if(side->best_price_bucket_index == SIZE_MAX) {
        return quantity;
    }
    if((direction * (side->best_price - limit_price)) < 0) {
        return quantity;
    }

    uint64_t unfilled_quantity = quantity;

    for(size_t i = 0; i < side->price_buckets_capacity; i++) {
        if(unfilled_quantity == 0) {
            break;
        }
        if(side->best_price_bucket_index == SIZE_MAX) {
            break;
        }
        if((direction * (side->best_price - limit_price)) < 0) {
            break;
        }

        LimitOrderBook_PriceBucket* price_bucket = &side->price_buckets[side->best_price_bucket_index];

        size_t order_index = price_bucket->orders_head_index;
        for(size_t j = 0; j < price_bucket->orders_capacity; j++) {
            if(unfilled_quantity == 0) {
                break;
            }
            if(price_bucket->total_orders == 0) {
                break;
            }

            if(price_bucket->orders[order_index] > 0) {
                uint64_t filled_quantity = (unfilled_quantity < price_bucket->orders[order_index]) ? unfilled_quantity : price_bucket->orders[order_index];
                unfilled_quantity -= filled_quantity;
                price_bucket->orders[order_index] -= filled_quantity;
                price_bucket->total_quantity -= filled_quantity;

                if(price_bucket->orders[order_index] == 0) {
                    price_bucket->total_orders--;
                }
            }

            order_index++;
            if(order_index >= price_bucket->orders_capacity) {
                order_index -= price_bucket->orders_capacity;
            }
        }

        for(size_t j = 0; j < price_bucket->order_references_capacity; j++) {
            LimitOrderBook_OrderReference* order_reference = &price_bucket->order_references[j];
            if((order_reference->price_direction != 0) && (price_bucket->orders[order_reference->order_index] == 0)) {
                for(size_t k = 0; k < limit_order_book->total_trader_accounts; k++) {
                    LimitOrderBook_TraderAccount* trader_account = &limit_order_book->trader_accounts[k];
                    for(size_t l = 0; l < trader_account->order_references_capacity; l++) {
                        if(trader_account->order_references[l] == order_reference) {
                            trader_account->order_references[l] = NULL;
                            trader_account->total_order_references--;
                            if(trader_account->total_order_references == 0) {
                                trader_account->order_references_head_index = SIZE_MAX;
                                trader_account->order_references_tail_index = SIZE_MAX;
                            }
                        }
                    }
                }
                destroy_limit_order_book_order_reference(order_reference);
            }
        }

        if(price_bucket->total_orders == 0) {
            price_bucket->orders_head_index = SIZE_MAX;
            price_bucket->orders_tail_index = SIZE_MAX;
            price_bucket->order_references_head_index = SIZE_MAX;
            price_bucket->order_references_tail_index = SIZE_MAX;

            size_t empty_worst_price_offset = (size_t)((direction * (side->best_price - side->worst_price)) / tick_size);
            if(empty_worst_price_offset == 0) {
                side->best_price = INT64_MAX;
                side->worst_price = INT64_MAX;
                side->best_price_bucket_index = SIZE_MAX;
            }
            else {
                for(size_t j = 1; j <= empty_worst_price_offset; j++) {
                    size_t index = side->best_price_bucket_index + j;
                    if(index >= side->price_buckets_capacity) {
                        index -= side->price_buckets_capacity;
                    }
                    if(side->price_buckets[index].total_orders > 0) {
                        side->best_price_bucket_index = index;
                        side->best_price -= direction * tick_size * ((int64_t)j);
                        break;
                    }
                }
            }
        }
        else {
            for(size_t j = 0; j < price_bucket->orders_capacity; j++) {
                if(price_bucket->orders[price_bucket->orders_head_index] > 0) {
                    break;
                }
                price_bucket->orders_head_index++;
                if(price_bucket->orders_head_index >= price_bucket->orders_capacity) {
                    price_bucket->orders_head_index -= price_bucket->orders_capacity;
                }
            }
        }
    }
    return unfilled_quantity;
}

uint64_t match_limit_order_book_order_using_pro_rata(LimitOrderBook* limit_order_book, int8_t price_direction, int64_t limit_price, uint64_t quantity) {
    LimitOrderBook_Side* side = (price_direction == LIMIT_ORDER_BOOK_BID) ? &limit_order_book->bids : &limit_order_book->asks;
    int64_t tick_size = limit_order_book->tick_size;
    int64_t direction = price_direction;

    if(side->best_price_bucket_index == SIZE_MAX) {
        return quantity;
    }
    if((direction * (side->best_price - limit_price)) < 0) {
        return quantity;
    }

    uint64_t unfilled_quantity = quantity;

    for(size_t i = 0; i < side->price_buckets_capacity; i++) {
        if(unfilled_quantity == 0) {
            break;
        }
        if(side->best_price_bucket_index == SIZE_MAX) {
            break;
        }
        if((direction * (side->best_price - limit_price)) < 0) {
            break;
        }

        LimitOrderBook_PriceBucket* price_bucket = &side->price_buckets[side->best_price_bucket_index];
        uint64_t total_quantity_in_bucket = price_bucket->total_quantity;

        if(total_quantity_in_bucket <= unfilled_quantity) {
            unfilled_quantity -= total_quantity_in_bucket;
            price_bucket->total_quantity = 0;

            for(size_t j = 0; j < price_bucket->orders_capacity; j++) {
                price_bucket->orders[j] = 0;
            }

            price_bucket->total_orders = 0;
            price_bucket->orders_head_index = SIZE_MAX;
            price_bucket->orders_tail_index = SIZE_MAX;
        }
        else {
            uint64_t starting_unfilled_quantity = unfilled_quantity;
            uint64_t starting_bucket_quantity = total_quantity_in_bucket;

            for(size_t j = 0; j < price_bucket->orders_capacity; j++) {
                if(price_bucket->orders[j] > 0) {
                    uint64_t fill_quantity = (starting_unfilled_quantity * price_bucket->orders[j]) / starting_bucket_quantity;
                    if(fill_quantity > price_bucket->orders[j]) {
                        fill_quantity = price_bucket->orders[j];
                    }
                    if(fill_quantity > 0) {
                        price_bucket->orders[j] -= fill_quantity;
                        price_bucket->total_quantity -= fill_quantity;
                        unfilled_quantity -= fill_quantity;
                        if(price_bucket->orders[j] == 0) {
                            price_bucket->total_orders--;
                        }
                    }
                }
            }

            if(price_bucket->total_orders == 0) {
                price_bucket->orders_head_index = SIZE_MAX;
                price_bucket->orders_tail_index = SIZE_MAX;
            }
            else {
                for(size_t j = 0; j < price_bucket->orders_capacity; j++) {
                    if(price_bucket->orders[price_bucket->orders_head_index] > 0) {
                        break;
                    }
                    price_bucket->orders_head_index++;
                    if(price_bucket->orders_head_index >= price_bucket->orders_capacity) {
                        price_bucket->orders_head_index -= price_bucket->orders_capacity;
                    }
                }
                for(size_t j = 0; j < price_bucket->orders_capacity; j++) {
                    if(price_bucket->orders[price_bucket->orders_tail_index] > 0) {
                        break;
                    }
                    if(price_bucket->orders_tail_index == 0) {
                        price_bucket->orders_tail_index = price_bucket->orders_capacity - 1;
                    }
                    else {
                        price_bucket->orders_tail_index--;
                    }
                }
            }

            size_t order_index = price_bucket->orders_head_index;
            for(size_t j = 0; j < price_bucket->orders_capacity; j++) {
                if(unfilled_quantity == 0) {
                    break;
                }
                if(price_bucket->total_orders == 0) {
                    break;
                }

                if(price_bucket->orders[order_index] > 0) {
                    uint64_t fill_quantity = (unfilled_quantity < price_bucket->orders[order_index]) ? unfilled_quantity : price_bucket->orders[order_index];
                    unfilled_quantity -= fill_quantity;
                    price_bucket->orders[order_index] -= fill_quantity;
                    price_bucket->total_quantity -= fill_quantity;

                    if(price_bucket->orders[order_index] == 0) {
                        price_bucket->total_orders--;
                    }
                }

                order_index++;
                if(order_index >= price_bucket->orders_capacity) {
                    order_index -= price_bucket->orders_capacity;
                }
            }

            if(price_bucket->total_orders > 0) {
                for(size_t j = 0; j < price_bucket->orders_capacity; j++) {
                    if(price_bucket->orders[price_bucket->orders_head_index] > 0) {
                        break;
                    }
                    price_bucket->orders_head_index++;
                    if(price_bucket->orders_head_index >= price_bucket->orders_capacity) {
                        price_bucket->orders_head_index -= price_bucket->orders_capacity;
                    }
                }
            }
        }

        for(size_t j = 0; j < price_bucket->order_references_capacity; j++) {
            LimitOrderBook_OrderReference* order_reference = &price_bucket->order_references[j];
            if((order_reference->price_direction != 0) && (price_bucket->orders[order_reference->order_index] == 0)) {
                for(size_t k = 0; k < limit_order_book->total_trader_accounts; k++) {
                    LimitOrderBook_TraderAccount* trader_account = &limit_order_book->trader_accounts[k];
                    for(size_t l = 0; l < trader_account->order_references_capacity; l++) {
                        if(trader_account->order_references[l] == order_reference) {
                            trader_account->order_references[l] = NULL;
                            trader_account->total_order_references--;
                            if(trader_account->total_order_references == 0) {
                                trader_account->order_references_head_index = SIZE_MAX;
                                trader_account->order_references_tail_index = SIZE_MAX;
                            }
                        }
                    }
                }
                destroy_limit_order_book_order_reference(order_reference);
            }
        }

        if(price_bucket->total_orders == 0) {
            price_bucket->order_references_head_index = SIZE_MAX;
            price_bucket->order_references_tail_index = SIZE_MAX;

            size_t empty_worst_price_offset = (size_t)((direction * (side->best_price - side->worst_price)) / tick_size);
            if(empty_worst_price_offset == 0) {
                side->best_price = INT64_MAX;
                side->worst_price = INT64_MAX;
                side->best_price_bucket_index = SIZE_MAX;
            }
            else {
                for(size_t j = 1; j <= empty_worst_price_offset; j++) {
                    size_t index = side->best_price_bucket_index + j;
                    if(index >= side->price_buckets_capacity) {
                        index -= side->price_buckets_capacity;
                    }
                    if(side->price_buckets[index].total_orders > 0) {
                        side->best_price_bucket_index = index;
                        side->best_price -= direction * tick_size * ((int64_t)j);
                        break;
                    }
                }
            }
        }
    }

    return unfilled_quantity;
}

void add_order_to_limit_order_book(LimitOrderBook* limit_order_book, size_t trader_account_index, int8_t price_direction, int64_t price, uint64_t quantity) {
    const size_t resize_increment = 10;

    if(trader_account_index >= limit_order_book->total_trader_accounts) {
        size_t new_total_trader_accounts = trader_account_index + 1;
        LimitOrderBook_TraderAccount* new_trader_accounts = malloc(new_total_trader_accounts * sizeof(LimitOrderBook_TraderAccount));
        for(size_t i = 0; i < limit_order_book->total_trader_accounts; i++) {
            new_trader_accounts[i] = limit_order_book->trader_accounts[i];
        }
        for(size_t i = limit_order_book->total_trader_accounts; i < new_total_trader_accounts; i++) {
            new_trader_accounts[i] = create_limit_order_book_trader_account();
        }

        free(limit_order_book->trader_accounts);
        limit_order_book->total_trader_accounts = new_total_trader_accounts;
        limit_order_book->trader_accounts = new_trader_accounts;
    }
    LimitOrderBook_TraderAccount* trader_account = &limit_order_book->trader_accounts[trader_account_index];

    LimitOrderBook_Side* side = (price_direction == LIMIT_ORDER_BOOK_BID) ? &limit_order_book->bids : &limit_order_book->asks;
    size_t price_bucket_index = get_limit_order_book_price_bucket_index(side, limit_order_book->tick_size, price_direction, price);
    LimitOrderBook_PriceBucket* price_bucket = &side->price_buckets[price_bucket_index];

    if(price_bucket->orders == NULL) {
        price_bucket->orders_capacity = resize_increment;
        price_bucket->orders = malloc(price_bucket->orders_capacity * sizeof(uint64_t));
        for(size_t i = 0; i < price_bucket->orders_capacity; i++) {
            price_bucket->orders[i] = 0;
        }
    }
    if(price_bucket->order_references == NULL) {
        price_bucket->order_references_capacity = resize_increment;
        price_bucket->order_references = malloc(price_bucket->order_references_capacity * sizeof(LimitOrderBook_OrderReference));
        for(size_t i = 0; i < price_bucket->order_references_capacity; i++) {
            price_bucket->order_references[i] = create_limit_order_book_order_reference();
        }
    }

    size_t new_orders_tail_index;
    if(price_bucket->total_orders == 0) {
        new_orders_tail_index = 0;
        price_bucket->orders_head_index = 0;
    }
    else {
        new_orders_tail_index = price_bucket->orders_tail_index + 1;
        if(new_orders_tail_index >= price_bucket->orders_capacity) {
            new_orders_tail_index -= price_bucket->orders_capacity;
        }
        if(new_orders_tail_index == price_bucket->orders_head_index) {
            compact_orders_in_limit_order_book_price_bucket(price_bucket);

            if(price_bucket->total_orders == (uint64_t)price_bucket->orders_capacity) {
                size_t new_orders_capacity = price_bucket->orders_capacity + resize_increment;
                uint64_t* new_orders = malloc(new_orders_capacity * sizeof(uint64_t));

                size_t old_order_index = price_bucket->orders_head_index;
                for(size_t i = 0; i < price_bucket->orders_capacity; i++) {
                    new_orders[i] = price_bucket->orders[old_order_index];
                    old_order_index++;
                    if(old_order_index >= price_bucket->orders_capacity) {
                        old_order_index -= price_bucket->orders_capacity;
                    }
                }
                for(size_t i = price_bucket->orders_capacity; i < new_orders_capacity; i++) {
                    new_orders[i] = 0;
                }

                for(size_t i = 0; i < price_bucket->order_references_capacity; i++) {
                    LimitOrderBook_OrderReference* order_reference = &price_bucket->order_references[i];
                    if(order_reference->price_direction != 0) {
                        if(order_reference->order_index >= price_bucket->orders_head_index) {
                            order_reference->order_index -= price_bucket->orders_head_index;
                        }
                        else {
                            order_reference->order_index += price_bucket->orders_capacity - price_bucket->orders_head_index;
                        }
                    }
                }

                free(price_bucket->orders);
                price_bucket->orders_head_index = 0;
                price_bucket->orders_tail_index = price_bucket->orders_capacity - 1;
                price_bucket->orders_capacity = new_orders_capacity;
                price_bucket->orders = new_orders;
            }

            new_orders_tail_index = price_bucket->orders_tail_index + 1;
            if(new_orders_tail_index >= price_bucket->orders_capacity) {
                new_orders_tail_index -= price_bucket->orders_capacity;
            }
        }
    }

    size_t new_order_references_tail_index;
    if(price_bucket->total_orders == 0) {
        new_order_references_tail_index = 0;
        price_bucket->order_references_head_index = 0;
    }
    else {
        new_order_references_tail_index = price_bucket->order_references_tail_index + 1;
        if(new_order_references_tail_index >= price_bucket->order_references_capacity) {
            new_order_references_tail_index -= price_bucket->order_references_capacity;
        }
        if(new_order_references_tail_index == price_bucket->order_references_head_index) {
            size_t new_order_references_capacity = price_bucket->order_references_capacity;
            if((price_bucket->total_orders + 1) >= (uint64_t)new_order_references_capacity) {
                new_order_references_capacity += resize_increment;
            }
            LimitOrderBook_OrderReference* new_order_references = malloc(new_order_references_capacity * sizeof(LimitOrderBook_OrderReference));

            size_t old_order_reference_index = price_bucket->order_references_head_index;
            size_t new_order_reference_index = 0;
            for(size_t i = 0; i < price_bucket->order_references_capacity; i++) {
                LimitOrderBook_OrderReference* old_order_reference = &price_bucket->order_references[old_order_reference_index];
                if(old_order_reference->price_direction != 0) {
                    new_order_references[new_order_reference_index] = *old_order_reference;

                    for(size_t j = 0; j < limit_order_book->total_trader_accounts; j++) {
                        LimitOrderBook_TraderAccount* other_trader_account = &limit_order_book->trader_accounts[j];
                        for(size_t k = 0; k < other_trader_account->order_references_capacity; k++) {
                            if(other_trader_account->order_references[k] == old_order_reference) {
                                other_trader_account->order_references[k] = &new_order_references[new_order_reference_index];
                            }
                        }
                    }

                    new_order_reference_index++;
                }

                old_order_reference_index++;
                if(old_order_reference_index >= price_bucket->order_references_capacity) {
                    old_order_reference_index -= price_bucket->order_references_capacity;
                }
            }
            for(size_t i = new_order_reference_index; i < new_order_references_capacity; i++) {
                new_order_references[i] = create_limit_order_book_order_reference();
            }

            free(price_bucket->order_references);
            price_bucket->order_references_head_index = 0;
            price_bucket->order_references_tail_index = new_order_reference_index - 1;
            price_bucket->order_references_capacity = new_order_references_capacity;
            price_bucket->order_references = new_order_references;

            new_order_references_tail_index = price_bucket->order_references_tail_index + 1;
        }
    }

    price_bucket->orders[new_orders_tail_index] = quantity;
    price_bucket->orders_tail_index = new_orders_tail_index;

    LimitOrderBook_OrderReference* order_reference = &price_bucket->order_references[new_order_references_tail_index];
    order_reference->price_direction = price_direction;
    order_reference->price = price;
    order_reference->order_index = new_orders_tail_index;
    price_bucket->order_references_tail_index = new_order_references_tail_index;

    price_bucket->total_orders++;
    price_bucket->total_quantity += quantity;

    if(trader_account->order_references == NULL) {
        trader_account->order_references_capacity = resize_increment;
        trader_account->order_references = malloc(trader_account->order_references_capacity * sizeof(LimitOrderBook_OrderReference*));
        for(size_t i = 0; i < trader_account->order_references_capacity; i++) {
            trader_account->order_references[i] = NULL;
        }
    }

    size_t new_trader_account_order_references_tail_index;
    if(trader_account->total_order_references == 0) {
        new_trader_account_order_references_tail_index = 0;
        trader_account->order_references_head_index = 0;
    }
    else {
        new_trader_account_order_references_tail_index = trader_account->order_references_tail_index + 1;
        if(new_trader_account_order_references_tail_index >= trader_account->order_references_capacity) {
            new_trader_account_order_references_tail_index -= trader_account->order_references_capacity;
        }
        if(new_trader_account_order_references_tail_index == trader_account->order_references_head_index) {
            size_t new_trader_account_order_references_capacity = trader_account->order_references_capacity;
            if((trader_account->total_order_references + 1) >= new_trader_account_order_references_capacity) {
                new_trader_account_order_references_capacity += resize_increment;
            }
            LimitOrderBook_OrderReference** new_trader_account_order_references = malloc(new_trader_account_order_references_capacity * sizeof(LimitOrderBook_OrderReference*));

            size_t old_trader_account_order_reference_index = trader_account->order_references_head_index;
            size_t new_trader_account_order_reference_index = 0;
            for(size_t i = 0; i < trader_account->order_references_capacity; i++) {
                if(trader_account->order_references[old_trader_account_order_reference_index] != NULL) {
                    new_trader_account_order_references[new_trader_account_order_reference_index] = trader_account->order_references[old_trader_account_order_reference_index];
                    new_trader_account_order_reference_index++;
                }

                old_trader_account_order_reference_index++;
                if(old_trader_account_order_reference_index >= trader_account->order_references_capacity) {
                    old_trader_account_order_reference_index -= trader_account->order_references_capacity;
                }
            }
            for(size_t i = new_trader_account_order_reference_index; i < new_trader_account_order_references_capacity; i++) {
                new_trader_account_order_references[i] = NULL;
            }

            free(trader_account->order_references);
            trader_account->order_references_head_index = 0;
            trader_account->order_references_tail_index = new_trader_account_order_reference_index - 1;
            trader_account->order_references_capacity = new_trader_account_order_references_capacity;
            trader_account->order_references = new_trader_account_order_references;

            new_trader_account_order_references_tail_index = trader_account->order_references_tail_index + 1;
        }
    }

    trader_account->order_references[new_trader_account_order_references_tail_index] = order_reference;
    trader_account->order_references_tail_index = new_trader_account_order_references_tail_index;
    trader_account->total_order_references++;
}

void cancel_order_in_limit_order_book(LimitOrderBook* limit_order_book, size_t trader_account_index, size_t order_reference_index) {
    LimitOrderBook_TraderAccount* trader_account = &limit_order_book->trader_accounts[trader_account_index];
    LimitOrderBook_OrderReference* order_reference = trader_account->order_references[order_reference_index];
    if(order_reference == NULL) {
        return;
    }

    int8_t price_direction = order_reference->price_direction;
    int64_t price = order_reference->price;
    size_t order_index = order_reference->order_index;

    LimitOrderBook_Side* side = (price_direction == LIMIT_ORDER_BOOK_BID) ? &limit_order_book->bids : &limit_order_book->asks;
    int64_t tick_size = limit_order_book->tick_size;
    int64_t direction = price_direction;
    size_t price_bucket_index = get_limit_order_book_price_bucket_index(side, tick_size, price_direction, price);
    LimitOrderBook_PriceBucket* price_bucket = &side->price_buckets[price_bucket_index];

    uint64_t quantity = price_bucket->orders[order_index];

    price_bucket->orders[order_index] = 0;
    destroy_limit_order_book_order_reference(order_reference);

    trader_account->order_references[order_reference_index] = NULL;
    trader_account->total_order_references--;
    if(trader_account->total_order_references == 0) {
        trader_account->order_references_head_index = SIZE_MAX;
        trader_account->order_references_tail_index = SIZE_MAX;
    }

    price_bucket->total_orders--;
    price_bucket->total_quantity -= quantity;

    if(price_bucket->total_orders == 0) {
        price_bucket->orders_head_index = SIZE_MAX;
        price_bucket->orders_tail_index = SIZE_MAX;
        price_bucket->order_references_head_index = SIZE_MAX;
        price_bucket->order_references_tail_index = SIZE_MAX;

        size_t worst_price_offset = (size_t)((direction * (side->best_price - side->worst_price)) / tick_size);

        if(price == side->best_price) {
            if(worst_price_offset == 0) {
                side->best_price = INT64_MAX;
                side->worst_price = INT64_MAX;
                side->best_price_bucket_index = SIZE_MAX;
            }
            else {
                for(size_t i = 0; i <= worst_price_offset; i++) {
                    size_t index = side->best_price_bucket_index + i;
                    if(index >= side->price_buckets_capacity) {
                        index -= side->price_buckets_capacity;
                    }
                    if(side->price_buckets[index].total_orders > 0) {
                        side->best_price_bucket_index = index;
                        side->best_price -= (direction * tick_size * ((int64_t)i));
                        break;
                    }
                }
            }
        }
        else if(price == side->worst_price) {
            for(size_t i = worst_price_offset; i > 0; i--) {
                size_t index = side->best_price_bucket_index + (i - 1);
                if(index >= side->price_buckets_capacity) {
                    index -= side->price_buckets_capacity;
                }
                if(side->price_buckets[index].total_orders > 0) {
                    side->worst_price = side->best_price - (direction * tick_size * ((int64_t)(i - 1)));
                    break;
                }
            }
        }
    }
    else {
        for(size_t i = 0; i < price_bucket->orders_capacity; i++) {
            if(price_bucket->orders[price_bucket->orders_head_index] > 0) {
                break;
            }
            price_bucket->orders_head_index++;
            if(price_bucket->orders_head_index >= price_bucket->orders_capacity) {
                price_bucket->orders_head_index -= price_bucket->orders_capacity;
            }
        }
        for(size_t i = 0; i < price_bucket->orders_capacity; i++) {
            if(price_bucket->orders[price_bucket->orders_tail_index] > 0) {
                break;
            }
            if(price_bucket->orders_tail_index == 0) {
                price_bucket->orders_tail_index = price_bucket->orders_capacity - 1;
            }
            else {
                price_bucket->orders_tail_index--;
            }
        }
    }
}

void shrink_limit_order_book(LimitOrderBook* limit_order_book) {
    LimitOrderBook_Side* bids = &limit_order_book->bids;
    if(bids->best_price_bucket_index == SIZE_MAX) {
        destroy_limit_order_book_side(bids);
    }
    else {
        size_t worst_price_offset = (size_t)((LIMIT_ORDER_BOOK_BID * (bids->best_price - bids->worst_price)) / limit_order_book->tick_size);
        rebuild_limit_order_book_side(bids, worst_price_offset + 1);
        for(size_t i = 0; i < bids->price_buckets_capacity; i++) {
            compact_orders_in_limit_order_book_price_bucket(&bids->price_buckets[i]);
        }
    }

    LimitOrderBook_Side* asks = &limit_order_book->asks;
    if(asks->best_price_bucket_index == SIZE_MAX) {
        destroy_limit_order_book_side(asks);
    }
    else {
        size_t worst_price_offset = (size_t)((LIMIT_ORDER_BOOK_ASK * (asks->best_price - asks->worst_price)) / limit_order_book->tick_size);
        rebuild_limit_order_book_side(asks, worst_price_offset + 1);
        for(size_t i = 0; i < asks->price_buckets_capacity; i++) {
            compact_orders_in_limit_order_book_price_bucket(&asks->price_buckets[i]);
        }
    }
}

int main() {
    printf("Limit Order Book");
}
