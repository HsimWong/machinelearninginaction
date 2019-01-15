Machine Learning in Action 
==========================
Updated on Jan. 15, 2019

This forked repo was first implemented in Python 2.6 and is right now working poorly with python 3.x, which is going to be re-implemented in Python 3.5 in order to help readers get more help when understanding code lines.

The original source code is to go with "Machine Learning in Action" 
by Peter Harrington published by Manning Inc.
The official page for this book can be found here: http://manning.com/pharrington/

Some algorithms in newly implementation is optimized in the facet of complexity. For example, simply using "Sorting" in finding the nearest points and major votes are considered to be auxilury, with a complexity of O(nlogn). Newly implementation used a queue and simply scanned the dataset and pushed the proper items into the queue, only with a complexity of O(n).

All the code examples in ./Python2_implementation were working on Python 2.6, there shouldn't be any problems with the 2.7.  NumPy will be needed for most examples.  If you have trouble running any of the examples us know on the Forum for this book: http://www.manning-sandbox.com/forum.jspa?forumID=728.  
