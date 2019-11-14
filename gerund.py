# Gerund: a joyful language for speech dictation.
#
# Voice dictation seems to work well on gerunds,
# so most builtin words end in -ing.
#
# See http://tunes.org/~iepos/joy.html for lots of combinators.
#
# Your Working Directory (pwd) will be used to store definitions in *.ing files.
#
# $ python  gerund.py test
# $ python  gerund.py list
# $ python  gerund.py definitions
#
# $ python  gerund.py 'define incr 1 adding '  ' 3 9 adding incr '
#
# $ python  gerund.py 'define incr: 1 adding.' 'define double: duplicating adding.' '3 double 9 double adding incr'
# >>>[25.0]

import glob
import os
import os.path
import re
import sys

import logging
import traceback

Debug = False

NONALFA = re.compile('[^A-Za-z0-9]+')
NUMBER = re.compile('^[0-9]+([.][0-9]+)?$')

STOPS = set(['', 'the', 'a', 'an', 'number', 'also'])

def Chop(s):
  t = NONALFA.sub(' ', s).lower()
  return [e for e in t.split() if e]

ORDINALS = dict(
  first=1, second=2, third=3, fourth=4, forth=4, fifth=5,
  sixth=6, seventh=7, eighth=8, nineth=9, tenth=10,
  eleventh=11, twelfth=12,
)
ORDINALS['1st'] = 1
ORDINALS['2nd'] = 2
ORDINALS['ii'] = 2
ORDINALS['3rd'] = 3
ORDINALS['iii'] = 3
ORDINALS['4th'] = 4
ORDINALS['iv'] = 4
ORDINALS['for'] = 4
ORDINALS['5th'] = 5
ORDINALS['6th'] = 6
ORDINALS['7th'] = 7
ORDINALS['8th'] = 8
ORDINALS['9th'] = 9
ORDINALS['10th'] = 10
ORDINALS['11th'] = 11
ORDINALS['12th'] = 12


class Gerund(object):
  def __init__(self, dirname=None, ticks=1000000, depth=100):
    self.dirname = dirname
    self.max_ticks = ticks  # Max steps
    self.max_depth = depth  # Recusion
    self.Reset()
    self.depth = depth  # Recusion
    self.stack = []
    self.words = {}
    if self.dirname:
      self.Scan()

  def Scan(self):
    for ingfile in glob.glob(os.path.join(self.dirname, "*.ing")):
      word, ing = os.path.basename(ingfile).split('.')
      with open(ingfile) as fd:
        guts = fd.read()
        try:
          #logging.info('Compiling %s = %s', word, guts)
          cc = self.Compile(Chop(guts))
          #logging.info('Compiled %s = %s', word, cc)
          f = self.LambdaToEvalCompiledWords(cc)
          setattr(self, word, f)
          self.words[word] = cc
        except Exception as ex:
          logging.error('CANNOT Compile: %s ;;; %s', word, ex)
          logging.error('%s', traceback.format_exc(20))
        except:
          logging.error('CANNOT Compile: %s ;;; funny exception', word)

  def Store(self, word, vec):
    filename = os.path.join(self.dirname, '%s.ing' % word)
    with open(filename, 'w') as fd:
      print >>fd, ' '.join(vec)

  def LambdaToEvalCompiledWords(self, cc):
    return lambda: self.Eval(cc)

  def Reset(self):
    self.ticks = self.max_ticks  # Max steps
    self.depth = self.max_depth  # Recusion

  def Run(self, s):
    try:
      s = NONALFA.sub(' ', s)
      s = s.lower()
      ww = [str(x) for x in s.split(' ') if x not in STOPS]
      #logging.info('WORDS = %s', ww)
  
      if len(ww) > 1 and ww[0]=='define':
        rest = ww[1:]
        what, numWhatSlots = self.Compile(rest, just_one_word=True)
  
        if type(what) is not str:
          raise Exception('Not defining a word: %s' % repr(what))
  
        rest = rest[numWhatSlots:]
        cc = self.Compile(rest)
        setattr(self, what, lambda: self.Eval(cc))
  
        if self.dirname: self.Store(what, rest)
        self.words[what] = repr(rest)
  
        return ( what, cc )
  
      elif len(ww) == 1 and ww[0]=='list':
        return ' '.join([k for k in sorted(self.words)])
  
      elif len(ww) > 1 and ww[0]=='list':
        what, numWhatSlots = self.Compile(ww[1:], just_one_word=True)
        return ( what, self.words.get(what) )
  
      elif len(ww) == 1 and ww[0]=='definitions':
        return [pair for pair in sorted(self.words.items())]
  
      elif len(ww) == 1 and ww[0]=='test':
        return Test()

      elif len(ww) > 1 and ww[0]=='must':
        want = float(ww[1])
        self.stack = []
        self.Eval(self.Compile(ww[2:]))
        if len(self.stack) != 1:
          raise Exception('Stack length: want 1 got %d ---- %s' % (len(self.stack), self.stack))
        got = float(self.stack[-1])
        if want != got:
          raise Exception('"must" Failed: want %s got %s' % (want, got))
        return None
  
      else:
        self.Reset()
        self.stack = []
        comp = self.Compile(ww)
        #logging.info('COMPILE = %s', comp)
        self.Eval(comp)
        #logging.info('STACK = %s', self.stack)
        return tuple(self.stack + ['ticks=%d' % (self.max_ticks - self.ticks)] )
    except Exception as ex:
      traceback.print_exc()
      raise ex

  def Compile(self, ww, just_one_word=False):
    z = []
    i = 0
    while i < len(ww):
      w = ww[i]
      if type(w) != str:
        z.append(w)
      elif NUMBER.match(w):
        z.append(float(w))
      elif w == 'opening':
        vecs = [ [] ]
        i+=1
        while i < len(ww):
          w = ww[i]
          if w == 'closing':
            if len(vecs) < 2:
              # finish the loop.
              break
            else:
              tmp = vecs.pop()
              vecs[-1].append(self.Compile(tmp))
          elif w == 'opening':
            vecs.append([])
          else:
            vecs[-1].append(w)
          i+=1
        z.append(self.Compile(vecs[0]))
      else:
        if i+1 < len(ww) and type(ww[i+1]) is str:
          # Ordinals are special; they append to previous word.
          # i.e. "foo second" becomes "foo2".
          if ww[i+1] in ORDINALS:
            i+=1
            w += str(ORDINALS[ww[i]])

          # This has become a problem on Wed Mar 26 2014
          elif ww[i+1] in ["ng", "ing"]:
            i+=1
            w += "ing"

        # make compound words with 'with'.
        if i+2 < len(ww) and ww[i+1] == 'with' and type(ww[i+2]) is str:
          i+=2
          w += '_with_' + ww[i]

        if Debug: print '=== word = ', w
        z.append(w)
      i+=1
      if just_one_word:
        return z[0], i
    return z

  def In(self):
    self.depth -= 1
    if self.depth < 1:
      raise Exception("Too Deep")
  def Out(self):
    self.depth += 1
  def Tick(self):
    self.ticks -= 1
    if self.ticks < 1:
      raise Exception("Too many Ticks")

  def Eval(self, ww):
    self.In()
    self.Tick()
    i = 0
    while i < len(ww):
      self.Tick()
      w = ww[i]
      if Debug: print '<<< <<< <<<' + repr(w)
      if callable(w):  # obsolete?
        w()
      elif type(w) is str:
        f = getattr(self, w, None)
        if not f:
          raise Exception('Unknown word: %s' % w)
        f()
      else:
        # A literal.  Push it on the stack.
        self.stack.append(w)

      if Debug: print '>>> >>> >>>' + repr(self.stack)
      i+=1
    self.Out()


  def BinaryOp(self, op):
    x = self.stack.pop()
    y = self.stack.pop()
    self.stack.append(self.RunBinaryOp(y, x, op))

  def RunBinaryOp(self, y, x, op):
    if type(y) == list:
      if type(x) == list:
        if len(y) != len(x):
          raise Exception("RunBinaryOp with differnt len lists: %d vs %d" % (len(y), len(x)))
        return [self.RunBinaryOp(y1, x1, op) for y1, x1 in zip(y, x)]
      else:
        return [self.RunBinaryOp(y1, x, op) for y1 in y]
    elif type(x) == list:
        return [self.RunBinaryOp(y, x1, op) for x1 in x]
    else:
      return op(y, x)

  def UnaryOp(self, op):
    x = self.stack.pop()
    self.stack.append(self.RunUnaryOp(x, op))

  def RunUnaryOp(self, x, op):
    if type(x) == list:
      return [self.RunUnaryOp(x1, op) for x1 in x]
    else:
      return op(x)

  def negating(self):
    self.UnaryOp(lambda x: -x)

  def denying(self):
    self.UnaryOp(lambda x: int(not bool(x)))

  def adding(self):
    self.BinaryOp(lambda y, x: y + x)

  def subtracting(self):
    self.BinaryOp(lambda y, x: y - x)

  def multiplying(self):
    self.BinaryOp(lambda y, x: y * x)

  def dividing(self):
    self.BinaryOp(lambda y, x: y / x)

  def moding(self):
    self.BinaryOp(lambda y, x: y % x)
  def modding(self):
    self.BinaryOp(lambda y, x: y % x)
  def modulo(self):
    self.BinaryOp(lambda y, x: y % x)

  def equals(self):
    self.BinaryOp(lambda y, x: y == x)
  def equaling(self):
    self.BinaryOp(lambda y, x: y == x)

  def exceeds(self):
    self.BinaryOp(lambda y, x: y > x)
  def exceeding(self):
    self.BinaryOp(lambda y, x: y > x)

  def zero(self): self.stack.append(0.0)
  def won(self): self.stack.append(1.0)
  def one(self): self.stack.append(1.0)
  def two(self): self.stack.append(2.0)
  def to(self): self.stack.append(2.0)
  def too(self): self.stack.append(2.0)
  def three(self): self.stack.append(3.0)
  def four(self): self.stack.append(4.0)
  def five(self): self.stack.append(5.0)
  def six(self): self.stack.append(6.0)
  def sex(self): self.stack.append(6.0)
  def seven(self): self.stack.append(7.0)
  def aight(self): self.stack.append(8.0)
  def ate(self): self.stack.append(8.0)
  def eight(self): self.stack.append(8.0)
  def nine(self): self.stack.append(9.0)
  def ten(self): self.stack.append(10.0)
  def eleven(self): self.stack.append(11.0)
  def twelve(self): self.stack.append(12.0)

  def duplicating(self):
    self.stack.append(self.stack[-1])
  def duplicating1(self):
    self.stack.append(self.stack[-1])
  def duplicating2(self):
    self.stack.append(self.stack[-2])
    self.stack.append(self.stack[-2])
  def duplicating3(self):
    self.stack.append(self.stack[-3])
    self.stack.append(self.stack[-3])
    self.stack.append(self.stack[-3])
  def duplicating4(self):
    self.stack.append(self.stack[-4])
    self.stack.append(self.stack[-4])
    self.stack.append(self.stack[-4])
    self.stack.append(self.stack[-4])
  def duplicating5(self):
    self.stack.append(self.stack[-5])
    self.stack.append(self.stack[-5])
    self.stack.append(self.stack[-5])
    self.stack.append(self.stack[-5])
    self.stack.append(self.stack[-5])

  def sizing(self):
    tmp = self.stack.pop()
    self.stack.append(len(tmp))

  def counting(self):
    t = self.stack.pop()
    self.stack.append(list(range(1, 1+int(t))))

  def filtering(self):
    pred = self.stack.pop()
    vec = self.stack.pop()
    zz = []
    for x in vec:
      self.stack.append(x)
      self.Eval(pred)
      c = self.stack.pop()
      if c:
        zz.append(x)
    self.stack.append(zz)

  def mapping(self):
    code = self.stack.pop()
    vec = self.stack.pop()
    zz = []
    for x in vec:
      self.stack.append(x)
      self.Eval(code)
      zz.append(self.stack.pop())
    self.stack.append(zz)

  def reducing(self):
    code = self.stack.pop()
    current = self.stack.pop()
    vec = self.stack.pop()
    for x in vec:
      self.stack.append(current)
      self.stack.append(x)
      self.Eval(code)
      current = self.stack.pop()
    self.stack.append(current)

  def dropping(self):
    bools = self.stack.pop()
    items = self.stack.pop()
    z = [x for x, y in zip(items, bools) if y]
    self.stack.append(z)

  def choosing(self):
    t1 = self.stack.pop()
    t2 = self.stack.pop()
    self.stack.append(t2[int(t1) - 1])

  def choosing1(self):
    t = self.stack.pop()
    self.stack.append(t[0])
  def choosing2(self):
    t = self.stack.pop()
    self.stack.append(t[1])
  def choosing3(self):
    t = self.stack.pop()
    self.stack.append(t[2])
  def choosing4(self):
    t = self.stack.pop()
    self.stack.append(t[3])
  def choosing5(self):
    t = self.stack.pop()
    self.stack.append(t[4])
  def choosing6(self):
    t = self.stack.pop()
    self.stack.append(t[5])
  def choosing7(self):
    t = self.stack.pop()
    self.stack.append(t[6])
  def choosing8(self):
    t = self.stack.pop()
    self.stack.append(t[7])
  def choosing9(self):
    t = self.stack.pop()
    self.stack.append(t[8])

  def changing(self):
    t1 = self.stack.pop()
    t2 = self.stack.pop()
    t3 = self.stack.pop()
    t3[int(t2)] = t1

  def changing1(self):
    t1 = self.stack.pop()
    t3 = self.stack.pop()
    t3[0] = t1
  def changing2(self):
    t1 = self.stack.pop()
    t3 = self.stack.pop()
    t3[1] = t1
  def changing3(self):
    t1 = self.stack.pop()
    t3 = self.stack.pop()
    t3[2] = t1
  def changing4(self):
    t1 = self.stack.pop()
    t3 = self.stack.pop()
    t3[3] = t1
  def changing5(self):
    t1 = self.stack.pop()
    t3 = self.stack.pop()
    t3[4] = t1
  def changing6(self):
    t1 = self.stack.pop()
    t3 = self.stack.pop()
    t3[5] = t1
  def changing7(self):
    t1 = self.stack.pop()
    t3 = self.stack.pop()
    t3[6] = t1
  def changing8(self):
    t1 = self.stack.pop()
    t3 = self.stack.pop()
    t3[7] = t1
  def changing9(self):
    t1 = self.stack.pop()
    t3 = self.stack.pop()
    t3[8] = t1

  def getting1(self):
    self.stack.append(self.stack[-1])
  def getting2(self):
    self.stack.append(self.stack[-2])
  def getting3(self):
    self.stack.append(self.stack[-3])
  def getting4(self):
    self.stack.append(self.stack[-4])
  def getting5(self):
    self.stack.append(self.stack[-5])

  def putting1(self):
    t = self.stack.pop()
    self.stack[-1] = t
  def putting2(self):
    t = self.stack.pop()
    self.stack[-2] = t
  def putting3(self):
    t = self.stack.pop()
    self.stack[-3] = t
  def putting4(self):
    t = self.stack.pop()
    self.stack[-4] = t
  def putting5(self):
    t = self.stack.pop()
    self.stack[-5] = t

  def popping(self):
    self.stack.pop()
  def popping1(self):
    self.stack.pop()
  def popping2(self):
    self.stack.pop()
    self.stack.pop()
  def popping3(self):
    self.stack.pop()
    self.stack.pop()
    self.stack.pop()
  def popping4(self):
    self.stack.pop()
    self.stack.pop()
    self.stack.pop()
    self.stack.pop()
  def popping5(self):
    self.stack.pop()
    self.stack.pop()
    self.stack.pop()
    self.stack.pop()
    self.stack.pop()

  def swapping(self):
    t1 = self.stack.pop()
    t2 = self.stack.pop()
    self.stack.append(t1)
    self.stack.append(t2)

  def rotating(self):
    "rotate TOS, 2OS and 3OS, moving 3OS to TOS."  # http://corewar.co.uk/assembly/forth.htm
    t1 = self.stack.pop()
    t2 = self.stack.pop()
    t3 = self.stack.pop()
    self.stack.append(t2)
    self.stack.append(t1)
    self.stack.append(t3)

  def constructing(self):
    t1 = self.stack.pop()
    if type(t1) is not list:
      raise Exception('"constructing" expected a list on top of stack, got %s' % t1)
    t2 = self.stack.pop()
    t = [t2] + t1
    self.stack.append(t)

  def concatenating(self):
    t1 = self.stack.pop()
    if type(t1) is not list:
      raise Exception('"concatenating" expected a list on top of stack, got %s' % t1)
    t2 = self.stack.pop()
    if type(t2) is not list:
      raise Exception('"concatenating" expected a list second on stack, got %s' % t1)
    t = t2 + t1
    self.stack.append(t)

  def reducing_with_concatenating(self):
    t = self.stack.pop()
    if type(t) is not list:
      raise Exception('"reducing_with_concatenating" expected a list on top of stack, got %s' % t)
    z = []
    for e in t:
      z = z + e
    self.stack.append(z)

  def reducing_with_adding(self): self.summing()
  def something(self): self.summing()
  def summation(self): self.summing()
  def summarizing(self): self.summing()
  def summing(self):
    t = self.stack.pop()
    if type(t) is not list:
      raise Exception('"summing" expected a list on top of stack, got %s' % t)
    self.stack.append(sum(t))

  def reducing_with_multiplying(self): self.productizing()
  def productionizing(self): self.productizing()
  def productizing(self):
    t = self.stack.pop()
    if type(t) is not list:
      raise Exception('"productizing" expected a list on top of stack, got %s' % t)
    self.stack.append(reduce(lambda x,y: x*y, t, 1))

  def running(self): # i
    t = self.stack.pop()
    self.Eval(t)

  def listing(self): # i
    t = self.stack.pop()
    p1 = len(self.stack)
    self.Eval(t)
    p2 = len(self.stack)
    z = []
    p = p1
    while p < p2:
      z.append(self.stack[p])
      p += 1
    self.stack = self.stack[:p1]
    self.stack.append(z)

  def iterating(self): # i
    t = self.stack.pop()
    n = self.stack.pop()
    for _ in range(int(n)):
      self.Eval(t)

  def depending(self): # ifte
    f = self.stack.pop()
    t = self.stack.pop()
    c = self.stack.pop()
    if c:
      self.Eval(t)
    else:
      self.Eval(f)

  def dipping(self): # dip
    t1 = self.stack.pop()
    t2 = self.stack.pop()
    self.Eval(t1)
    self.stack.append(t2)
  def dipping2(self): # dip2
    t1 = self.stack.pop()
    t2 = self.stack.pop()
    t3 = self.stack.pop()
    self.Eval(t1)
    self.stack.append(t3)
    self.stack.append(t2)
  def dipping3(self): # dip3
    t1 = self.stack.pop()
    t2 = self.stack.pop()
    t3 = self.stack.pop()
    t4 = self.stack.pop()
    self.Eval(t1)
    self.stack.append(t4)
    self.stack.append(t3)
    self.stack.append(t2)

  def sipping(self): # sip
    t1 = self.stack.pop()
    t2 = self.stack.pop()
    self.stack.append(t2)
    self.Eval(t1)
    self.stack.append(t2)

  def repeating(self): # scalar|vec n repeating -> vec
    n = int(self.stack.pop())
    x = self.stack.pop()
    if type(x) is list:
      self.stack.append(n * x)
    else:
      self.stack.append(n * [x])

  def shifting(self): # vec i shifting -> vec # shift 0's in, slde everything right (positive i) or left (negative i) i places.
    i = int(self.stack.pop())
    vec = self.stack.pop()
    if type(vec) is not list:
      raise Exception('Not a list: %s' % vec)
    n = len(vec)
    if i > 0:
      i = min(i, n)
      j = n - i
      self.stack.append((i * [0]) + vec[:j])
    elif i < 0:
      i = min(-i, n)
      j = n - i
      self.stack.append(vec[-j:] + (i * [0]))
    else:
      self.stack.append(vec)
    
  def basis(self): # n basis -> vec # 1 in first positin, rest are zeroes.
    n = int(self.stack.pop())
    self.stack.append([1] + (n-1) * [0])
   
  def MarkingN(self, n):
    m = Marker(n)
    self.stack.append(m)

  def marking(self): self.MarkingN(0)
  def marking1(self): self.MarkingN(1)
  def marking2(self): self.MarkingN(2)
  def marking3(self): self.MarkingN(3)
  def marking4(self): self.MarkingN(4)
  def marking5(self): self.MarkingN(5)

  def FindMark(self):
    n = len(self.stack)
    for i in range(n):
      j = n - i - 1
      if type(self.stack[j]) is Marker:
        return j
    raise Exception("Mark not found: %s" % self.stack)
      
  def RetainingN(self, k):
    p = self.FindMark()
    n = self.stack[p].n
    j = p - n
    t = len(self.stack)
    for i in range(k):
      self.stack[j + i] = self.stack[t - k + i]
    self.stack = self.stack[:j+k]

  def retaining(self): self.RetainingN(0)
  def retaining1(self): self.RetainingN(1)
  def retaining2(self): self.RetainingN(2)
  def retaining3(self): self.RetainingN(3)
  def retaining4(self): self.RetainingN(4)
  def retaining5(self): self.RetainingN(5)

  def FetchingN(self, i):
    p = self.FindMark()
    x = self.stack[p-i]
    self.stack.append(x)
  def fetching1(self): self.FetchingN(1)
  def fetching2(self): self.FetchingN(2)
  def fetching3(self): self.FetchingN(3)
  def fetching4(self): self.FetchingN(4)
  def fetching5(self): self.FetchingN(5)

  def StoringN(self, i):
    x = self.stack.pop()
    p = self.FindMark()
    self.stack[p-i] = x
  def storing1(self): self.StoringN(1)
  def storing2(self): self.StoringN(2)
  def storing3(self): self.StoringN(3)
  def storing4(self): self.StoringN(4)
  def storing5(self): self.StoringN(5)

    


class Marker(object):
  def __init__(self, n):
    self.n = n
  def __repr__(self):
    return 'Marker(%d)' % self.n
  def __str__(self):
    return 'Marker(%d)' % self.n

def Test():

  if os.getenv('Debug'):
    Debug = True

  # TESTS
  t = Gerund()
  t.Run('define incr: 1 adding.')
  t.Run('must 8: 7 incr')
  t.Run('define double: duplicating adding.')
  t.Run('must 88: 44 double')
  t.Run('must 25: 3 double 9 double adding incr')
  t.Run('must 3: opening 44 55 66 closing sizing')
  t.Run('must 55: opening 44 55 66 closing 2 choosing')
  
  t.Run('must 0: 44 denying')
  t.Run('must 1: 0 denying')

  t.Run('must 3:    opening 111 222 333 closing  10 adding  sizing')
  t.Run('must 121:  opening 111 222 333 closing  10 adding  choosing1')
  t.Run('must 343:  opening 111 222 333 closing  10 adding  choosing3')

  t.Run('must 3:    opening 111 222 333 closing  opening 21 31 41 closing  adding  sizing')
  t.Run('must 132:  opening 111 222 333 closing  opening 21 31 41 closing  adding  choosing1')
  t.Run('must 374:  opening 111 222 333 closing  opening 21 31 41 closing  adding  choosing3')

  t.Run('must 3:    opening 111 222 333 closing  10 multiplying  sizing')
  t.Run('must 1110:  opening 111 222 333 closing  10 multiplying  choosing1')
  t.Run('must 3330:  opening 111 222 333 closing  10 multiplying  choosing3')

  t.Run('must 2: opening 44 opening 22 33 closing 66 closing 2 choosing sizing')
  t.Run('must 22: opening 44 opening 22 33 closing 66 closing 2 choosing 1 choosing')
  t.Run('must 33: opening 44 opening 22 33 closing 66 closing choosing2 choosing2')
  
  t.Run('must 33: 11 22 30 getting the third getting the third adding putting the third popping2')
  
  t.Run('must 42: 10 opening 30 2 adding closing running adding')
  t.Run('must 42: 10 opening 4 opening 5 3 adding closing running  multiplying closing running adding')
  t.Run('must 42: 8 10 opening 4 multiplying closing dipping adding')
  t.Run('must 42: 8 6 4 opening 4 multiplying closing dipping the second adding adding')
  t.Run('must 42: 8 6 3 1 opening 4 multiplying closing dipping the third adding adding adding')

  t.Run('must 4: 11 opening 22 33 44 closing constructing sizing')
  t.Run('must 110: 11 opening 22 33 44 closing constructing summing')

  t.Run('must 4: opening 11 closing opening 22 33 44 closing concatenating sizing')
  t.Run('must 110: opening 11 closing opening 22 33 44 closing concatenating summing')

  t.Run('must 720: opening 1 2 3 4 5 6 closing productizing ')
  t.Run('must 21: opening 1 2 3 4 5 6 closing summing ')
  t.Run('must 0: opening closing summing ')
  t.Run('must 121: opening 1 2 3 4 5 6 closing  100 opening adding closing  reducing')
  t.Run('must 121: 6 counting  100 opening adding closing  reducing')
  t.Run('must 100: opening closing  100 opening adding closing  reducing')

  t.Run('must 12: 6 counting  opening 2 modulo 0 equaling closing  filtering summing')

  t.Run('must 28: 7 counting summing')
  t.Run('must 140: 7 counting  opening duplicating multiplying closing mapping summing')

  # See http://tunes.org/~iepos/joy.html for the following identities.

  # i == dup dip zap
  t.Run('Define i111: duplicating dipping popping')
  t.Run('must 7: opening 3 4 adding closing  i111')
  # i == [[]] dip dip zap
  t.Run('Define i222: opening opening closing closing dipping dipping popping')
  t.Run('must 7: opening 3 4 adding closing  i222')
  # i == [[]] dip dip dip
  t.Run('Define i333: opening opening closing closing dipping dipping dipping')
  t.Run('must 7: opening 3 4 adding closing  i333')

  # unit == [] cons
  t.Run('Define unit111: opening closing constructing')
  # swap == unit dip
  t.Run('Define swap111:  unit111 dipping')
  t.Run('must 7: 3 10  swap111  subtracting')
  # dip == swap unit cat i
  t.Run('Define dip111: swapping unit111 concatenating running')
  t.Run('must 42: 8 10 opening 4 multiplying closing dip111 adding')

  # dup == [] sip
  t.Run('Define dup222:  opening closing sipping')
  t.Run('must 0:  17 dup222 subtracting')

  # dip == [[zap zap] sip i] cons sip
  t.Run('Define dip222:   opening opening popping popping closing sipping running closing constructing sipping')
  t.Run('must 42: 8 10 opening 4 multiplying closing dip222 adding')

  t.Run('Define prime:  duplicating counting opening getting2 swapping modulo 0 equaling closing mapping summing 2 equaling swapping popping')
  t.Run('must 1: 5 prime')
  t.Run('must 0: 6 prime')
  t.Run('must 1: 7 prime')
  t.Run('must 0: 8 prime')

  t.Run('Define squaring: duplicating multiplying')
  t.Run('Define distance2: marking the second: '
        '  fetching2 squaring , fetching1 squaring, adding;'
        '  retaining1 ')
  t.Run('must 25: 3 4 distance2 ')
  t.Run('must 121: 100 marking1; 6 counting '
        '  opening fetching1 adding storing1 0 closing, mapping;'
        '     fetching1 retaining1')

  return 'OKAY'

logging.basicConfig(level=logging.INFO)
if __name__ == '__main__':
  # Run command line args.
  h = Gerund(dirname='.')
  for a in sys.argv[1:]:
    print '<<< ' + repr(a)
    print '>>> ' + repr(h.Run(a))
