# gerund-lang
Gerund, a joyful language for Speech Dictation (like to Google Glass).

This is the standalone commandline version.

First chdir into the gerund-lang directory, because definitions are
loaded & stored as `*.ing` files in the working directory.

```
cd gerund-lang

python  gerund.py  test
python  gerund.py  list

python  gerund.py  fibonacci

python  gerund.py 'define incr 1 adding'  '3 9 adding incr'
```

( The version I actually used on Google Glass is in https://github.com/strickyak/mirror-happiness )
