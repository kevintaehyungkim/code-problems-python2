

#####################
### DECODE STRING ###
#####################
'''
Given an encoded string, return its decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. 
Note that k is guaranteed to be a positive integer.

Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. 
For example, there won't be input like 3a or 2[4].

Example 1:
Input: s = "3[a]2[bc]"
Output: "aaabcbc"

Example 2:
Input: s = "3[a2[c]]"
Output: "accaccacc"
'''

def decode_string(s):
        stack, curNum, curString = [], 0, ''
        for c in s:
            if c == '[':
                stack.append((curString, curNum))
                curString, curNum = '', 0
            elif c == ']':
                prevString, num = stack.pop()
                curString = prevString + num*curString
            elif c.isdigit():
                curNum = curNum*10 + int(c)
            else:
                curString += c
        return curString



###################
### DECODE WAYS ###
###################
'''
A message containing letters from A-Z is being encoded to numbers using the following mapping:

'A' -> 1
'B' -> 2
...
'Z' -> 26

Given a non-empty string containing only digits, determine the total number of ways to decode it.

Example 1:
Input: "12"
Output: 2

Example 2:
Input: "226"
Output: 3
'''

# key idea: recursion w memoization
# time: O(N)
# space O(N)
def numDecodings(s):
    if not s:
        return 0
    
    code_set = set([str(i) for i in range(1,27)])
    seen = {}
    
    def decode(i):
        if i == len(s):
            return 1
        
        if i in seen:
            return seen[i]
        
        if s[i] in code_set:
        
            if i == len(s)-1:
                return 1
            
            res = decode(i+1)
            
            if i+1 < len(s) and s[i:i+2] in code_set:
                res += decode(i + 2)
                
            seen[i] = res
            return res

        return 0
    
    return decode(0)


##################
### IS ANAGRAM ###
##################
'''
Given two strings s and t, write a function to determine if t is an anagram of s.

For example,
s = "anagram", t = "nagaram", return true.
s = "rat", t = "car", return false.

Note:
You may assume the string contains only lowercase alphabets.

Follow up:
What if the inputs contain unicode characters? How would you adapt your solution to such case?

Difficult: Easy
'''

# O(n) time
# O(1) space
def is_anagram(s,t):
	letter_dict = {}
	for letter in s:
		if letter in letter_dict:
			letter_dict[letter] = letter_dict[letter] + 1
		else:
			letter_dict[letter] = 1
	for letter in t:
		letter_dict[letter] = letter_dict[letter] - 1 
	for val in letter_dict.values():
		if val != 0:
			return False 
	return True


# O(nlogn) time
# O(1) space
def is_anagram(s,t):
	return sorted(s) == sorted(t)

if __name__ == '__main__':
  print is_anagram("anagram", "nagaram")



################################
### LONGEST UNIQUE SUBSTRING ###
################################
'''
Given a string, find the length of the longest substring without repeating characters.

Example 1:

Input: "abcabcbb"
Output: 3 
Explanation: The answer is "abc", with the length of 3. 

Example 2:

Input: "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.

Example 3:

Input: "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3. 
             
Note that the answer must be a substring, "pwke" is a subsequence and not a substring.

'''

def longest_unique_substring(s):
    maxWindow = 0
    start = end = 0
    alphabets = set()
    
    while( end < len(s) ):
        if( s[end] not in alphabets ):
            alphabets.add(s[end])
            end += 1
            maxWindow = max( maxWindow, end - start )
        else:
            alphabets.remove(s[start])
            start += 1
    return maxWindow




###################################
### PALINDROMIC SUBSTRING COUNT ###
###################################
'''
Given a string, your task is to count how many palindromic substrings in this string.

The substrings with different start indexes or end indexes are counted as different substrings even they consist of same characters.

Example 1:

Input: "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".
 

Example 2:

Input: "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
'''

# time: O(n^2)
# space: O(1)

class Solution(object):
    def countSubstrings(self, s):
        num_palindromes = 0
        for i in range(0,len(s)):
            num_palindromes += palindrome_helper(i, i, s)
            num_palindromes += palindrome_helper(i, i+1, s)
        return num_palindromes
              

def palindrome_helper(i1, i2, s):
    count = 0
    while i1 >= 0 and i2 < len(s) and s[i1]== s[i2]:
        i1 -= 1
        i2 += 1
        count += 1
        
    return count

    

#####################################
### LONGEST PALINDROMIC SUBSTRING ###
#####################################
'''
Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

Example 1:

Input: "babad"
Output: "bab"

Note: "aba" is also a valid answer.

Example 2:

Input: "cbbd"
Output: "bb"
'''

# time: O(N^2)
# space: O(1)

class Solution(object):
    
    def longestPalindrome(self, s):
        longest_palindrome = ""
        for i in range(0,len(s)):
            single = self.palindrome_helper(i, i, s)
            double = self.palindrome_helper(i, i+1, s)
            
            current_longest = max(single, double, key=len)
            if len(current_longest) > len(longest_palindrome):
                longest_palindrome = current_longest
            
        return longest_palindrome
              

    def palindrome_helper(self, i1, i2, s):
        curr = ""
        while i1 >= 0 and i2 < len(s) and s[i1]== s[i2]:
            curr = s[i1:i2+1]
            i1 -= 1
            i2 += 1

        return curr



