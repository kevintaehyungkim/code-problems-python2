#! /usr/bin/python

# def find_pairs(student_course_pairs):
# 	sids = set()
# 	sid_pairs = {} # tuple of every student id pair
# 	course_sids = {} # key - course id, value - set of sids

# 	# add student pairs and course/student map
# 	for p in student_course_pairs:
# 		sid, course = int(p[0]), p[1]

# 		# add student id to course
# 		if course in course_sids:
# 			course_sids[course].add(sid)
# 		else:
# 			course_sids[course] = {sid}

# 		# add student to student ids
# 		if sid not in sids:
# 			for sid2 in list(sids):
# 				s1, s2 = min(sid, sid2), max(sid,sid2)
# 				sid_pairs[(s1,s2)] = []
# 			sids.add(sid)

# 	# process student pairs
# 	for p in sid_pairs:
# 		print p
# 		sid1, sid2 = p[0], p[1]
# 		for c,s in course_sids.items():
# 			if sid1 in s and sid2 in s:
# 				sid_pairs[p].append(c)

# 	return sid_pairs


# student_course_pairs = [
#     ["58", "Software Design"],
#     ["58", "Linear Algebra"],
#     ["94", "Art History"],
#     ["94", "Operating Systems"],
#     ["17", "Software Design"],
#     ["58", "Mechanics"],
#     ["58", "Economics"],
#     ["17", "Linear Algebra"],
#     ["17", "Political Science"],
#     ["94", "Economics"]
# ]


def canCross(stones):
    last_stone = stones[-1]
    stoneSet = set(stones)
    visited = set()
    
    def goFurther(value,k):
    	print visited
        if (value+k not in stoneSet) or ((value,k) in visited):
            return False
        if value+k == last_stone:
            return True
        visited.add((value,k))
        if k > 2:
            return goFurther(value+k,k) or goFurther(value+k,k-1) or goFurther(value+k,k+1)
        return goFurther(value+k,k) or goFurther(value+k,k+1)
    
    return goFurther(stones[0],1)

# print(find_pairs(student_course_pairs))
print(canCross([0,1,3,4,5,7,9,10,12]))