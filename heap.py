import heapq


###############################
### K MOST FREQUENT ELEMENTS ###
###############################
'''
Given a non-empty array of integers, return the k most frequent elements.

Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Example 2:

Input: nums = [1], k = 1
Output: [1]

Note:

You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
Your algorithm's time complexity must be better than O(n log n), where n is the array's size.
It's guaranteed that the answer is unique, in other words the set of the top k frequent elements is unique.
You can return the answer in any order.
'''

'''
Heap: smallest element is always at the root

heappush - push value item onto the heap
heappop - pop and return smallest item from the heap

'''
def top_k_frequent(nums, k):

	num_count = {}
	for num in k:
		num_count.get(num,0) + 1

	heap = []
    
    for k in num_count:
    	heapq.heappush(heap, (k, num_count[k]))
    	if len(heap) > k:
    		heapq.heappop(heap)

    return [i[1] for i in heap]




