### Things I learned while developing this project

------------
If you want git ignore to ignore some file that already exists:
1. Commit all pending changes, then run this command:
```
git rm -r --cached .
```
This removes everything from the index, then just run:
```
git add .
```
Commit it:
```
git commit -m ".gitignore is now working"
```
-----------


