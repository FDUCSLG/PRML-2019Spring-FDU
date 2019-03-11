# PRML-Spring19-Fudan
Course Website for PRML Spring 2019 at Fudan University

## Agreement of sharing your course work

By using this repository to submit your coursework, you agree to share your coursework with your peers during the course and publicly for following year students. If you want to keep anonymous you could create a new github account and email the teaching assistant to assign you a mask student id to anonymize your submission (e.g. 23333333333 rather than 16307130177).

## Coursework Guidelines

* You should use python3 as your programming language, we would recommand you to install anaconda which ships with the required packages for you to finish your coursework.
* You should use packages like `numpy`, `matplotlib`, `pandas` or `scipy`, if not specified, do not use other non-standard packages.
* Both English and Chinese are acceptable, and there will be no difference in terms of marking as long as you can make yourself clear with your report.
* Write clearly and succinctly. Highlight your answer and key result that is asked in the coursework requirements by using **Bold** font. You should take your own risk if you intentionally or unintentionally make the marking un-straightforward.
* Bonus mark (no more than 20%) will be considered if you make more in-depth exploration or further development that could in turn inspire the coursework to be a better one and show your understanding of the course material, this should only be the case given that you have already met the requirements.
* **Don't identify yourself in your source code or report**. We will use an automated script to collect your code and report for marking in order to hide your identify, so don't spoil it by writing down your student id or name in your report. The only identification is the name of the directory that contains your submission, however it won't be revealed during marking.
* Please use the [issue system](https://github.com/ichn-hu/PRML-Spring19-Fudan/issues) to ask questions about the coursework and course, use proper tags whenever possible, (e.g. `assignment-1`). In this case any questions answered by the instructor, TAs or others will also be valuable for other students.
* If you find any mistakes in the coursework or the course website itself (e.g. typos) you are encouraged to correct it with a pull request, however, don't mix this kind of changes with your coursework submission pull request as stated in the following section.
* For any feedback, please consider email the TAs first.


## Submission Guidelines

We assume that you are familiar with github and git in general, if not please search online and ask your friends for help, although we will give you some hints bellow.

For each assignment, the file should be structured like this


```
.
├── assignment-1/
│   ├── 16307130177/
│   │   ├── report.pdf
│   │   └── source.py
│   └── handout/
│       └── __init__.py
```

The `handout/` directory contains the facilities provided for you to accomplish the assignment, they will most likely be provided as python functions so that you could import it to your source code by adding the `..` to your python path (see [here]() for example).

The workflow of doing the coursework is like:

1. You fork this repository, and clone your forked repository to your local workplace.
2. When new assignment is released, first pull the updates in the original repository in your local cloned workplace, and create a directory with your student id (or mask student id) under the assignment directory, and only work in your own directory so that you won't conflict with others.
3. In order to make the repository clean and easy for marking, you are only allowed to commit `*.py` and `report.pdf` files for your submission, you should make your main procedure in `source.py` file.
