package util

import "container/list"

type entry[K comparable, V any] struct {
	key   K
	value V
}

type LRUCache[K comparable, V any] struct {
	cap   int
	ll    *list.List
	index map[K]*list.Element
}

func NewLRU[K comparable, V any](capacity int) *LRUCache[K, V] {
	if capacity < 1 {
		capacity = 1
	}
	return &LRUCache[K, V]{
		cap:   capacity,
		ll:    list.New(),
		index: make(map[K]*list.Element, capacity),
	}
}

func (c *LRUCache[K, V]) Get(key K) (V, bool) {
	if el, ok := c.index[key]; ok {
		c.ll.MoveToFront(el)
		return el.Value.(entry[K, V]).value, true
	}
	var zero V
	return zero, false
}

func (c *LRUCache[K, V]) Put(key K, value V) {
	if el, ok := c.index[key]; ok {
		el.Value = entry[K, V]{key, value}
		c.ll.MoveToFront(el)
		return
	}
	el := c.ll.PushFront(entry[K, V]{key, value})
	c.index[key] = el
	if c.ll.Len() > c.cap {
		c.evict()
	}
}

func (c *LRUCache[K, V]) evict() {
	el := c.ll.Back()
	if el == nil { return }
	c.ll.Remove(el)
	kv := el.Value.(entry[K, V])
	delete(c.index, kv.key)
}
