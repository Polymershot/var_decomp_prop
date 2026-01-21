# Summary

This python module was created to implement the Variance Decomposition Process(V.D.P.) as seen in Besley (1991) "Conditioning Diagonostics. Collinearity and Weak Data in Regression". Wiley.. I did not seem to find any implementation of this collinearity diagonistic tool meant for linear regression in Python so I decided to give it a go. 
What makes this tool different from collinearity measures such as VIF is that it makes the collinearity relationships more apparent. However, the process of discovering the multicollinearity is quite complicated and requires a little experimentation when it comes to trying to find all the collinear relationships.

# Remarks

One possible issue with the V.D.P. function is the compute vs. memory tradeoff. What I mean by this is that in the function, I'm doing a lot of column scaling and vector/matrix operations while utilizing broadcasting (<https://numpy.org/doc/2.3/user/basics.broadcasting.html>). 
The numpy article references cases where broadcasting can lead to large amounts of memory allocation. Based on what I've read so far, the main issue seems to be with intermediate storage of values as mentioned in (<https://stackoverflow.com/questions/47309818/when-broadcasting-is-a-bad-idea-numpy>). 
With broadcasting, I am kinda stretching the lower rank scalar/vector to match the larger rank vector/matrix in order to do some operation/transformation between them. However, at the moment, I currently am unable to identify the bottlenecks in my function yet.

