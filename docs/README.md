

# Documentation compilation
We followed this webpage to create the doc with sphinx: https://daler.github.io/sphinxdoc-test/includeme.html

## Create the doc structure
This is to be done the first time, then the next to upate it. 
1) Create a folder called simplicial-kuramoto-docs near the simplicial-kuramoto git folder (same level):
```mkdir simplicial-kuramoto-docs```
2) Clone the simplicial-kuramoto repo in subfolder html: 
```git clone git@github.com:arnaudon/simplicial-kuramoto.git html```
3) move into it: 
```cd html```
4) Switch branches to gh-pages:
```
git branch gh-pages
git symbolic-ref HEAD refs/heads/gh-pages  # auto-switches branches to gh-pages
rm .git/index
git clean -fdx
```

## Update the doc
1) in the main repo, do
```
cd docs
make html
```
this will update the files in simplicial-kuramoto-docs. 

2) To update the git at the same time as compiling, just do 
```
make full
```

Alternatively, you can update the git by hand using:
```
cd ../../simplicial-kuramoto-docs/html
git add .
git commit -m "rebuilt docs"
git push origin gh-pages
```
