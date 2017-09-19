---
title: Maximum Subarry
date: 2017-09-19
categories:
- LeetCode/Algorithms
tags: 
- Algorithms
description: Find the contiguous subarray within an array (containing at least one number) which has the largest sum.
---
Find the contiguous subarray within an array (containing at least one number) which has the largest sum.
**For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
the contiguous subarray [4,-1,2,1] has the largest sum = 6.**
**The Solution（C++）：**
```cpp
int maxSubArray(vector<int>& nums) {
       int n = nums.size();
       int max = nums[0];
       int sum = nums[0];
       if(n == 0 )
           return 0;
       else if(n == 1)
           return nums[0];
       else
           for(int i = 1; i < n; i++){
                if(sum < 0){
                    sum = nums[i];
                }else
                    sum += nums[i];
               
                if(sum > max)
                    max = sum;
           }
           return max;    
}
```
