Title: Coderbyte Challenge: Bracket Matcher
Date: 2014-06-01
Slug: coderbyte-challenge-bracket-matcher


*Originally published June 2014*

Recently I've been attempting the programming challenges on [Coderbyte](https://coderbyte.com/CodingArea/Challenges/). Coderbyte is a site that has a handful of questions that allots points given how 1) fast you complete the challenge and 2) how correct your answer is.

Once you solve a challenge correctly, it shows you solutions from other users, which might be its most addicting part. Typically I'll either scoff at someone's approach or look at their answer in amazement.

For one challenge though, I wasn't sure.

The challenge is:


> 
> Have the function BracketMatcher(str) take the str parameter being passed and return 1 if the brackets are correctly matched and each one is accounted for. Otherwise return 0. For example: if str is "(hello (world))", then the output should be 1, but if str is "((hello (world))" then the output should be 0 because the brackets do not correctly match up. Only "(" and ")" will be used as brackets. If str contains no brackets return 1.
> 

I was convinced the best way to approach this is recursively. I'll look for the innermost matching pair of parentheses, remove them, then call the function again with the slimmed down string. At the end, if there are any remaining parentheses without a companion, return the answer.

Everyone else went a different direction: loop through each character and for each `(` or `)` character found, add to a counter for either the left or right of the pair. At the end, if the counters are equal, return a 1. Otherwise, return a 0.

Here's my code:


```
function BracketMatcher(str) {

  //Finds '(', then a set in which everything is accepted but '(' and ')' 0 or more times, then ')' 
  var reg = /\([^()] {0,}\)/;
  var result;

  var bracket = function(str) {
    //Check to see if we've found parentheses pairs
    var matched_str = str.match(reg);

    //If no more, see if any stragglers are remaining
    if (matched_str == null) {
      if (str.match(/\(|\)/) == null) { result = 1; }
      else { result = 0; }
    }
    else {
      //Replace found parentheses pair with nothing and continue
      str = str.replace(reg, '');
      bracket(str);
    }
  }
  bracket(str);
  return result; 
}
```

Here is the code from user "mattlarcs" (who often has great solutions):


```
function BracketMatcher(str) { 
  var lP = 0;
  var rP = 0;
  for(var i=0;i lP) return 0;
 }
 if(rP === lP) return 1;
 return 0;
}
```

Let's see how these perform against one another. To do this, I need some text with lots of parentheses. Some Lisp sample code is at the bottom of the post.


I defined the Lisp text as `string` and pit the two functions against each other. Firing up Chrome's console:


```
var iterations = 100000;
console.time('Function #1');
for(var i = 0; i < iterations; i++ ){
    BracketMatcher(string);
};
console.timeEnd('Function #1')

console.time('Function #2');
for(var i = 0; i < iterations; i++ ){
    BracketMatcherII(string);
};
console.timeEnd('Function #2')

Function #1: 34702.743ms
Function #2: 2534.664ms
```

My solution is nearly 13 times as slow! I need to spend more time on why this is the case. I know both the `match` and `replace` functions I'm using are expensive and there is certainly some optimization that can be done.


In the meantime, it's interesting how drastic of a difference the performance of the two are.



### Lisp syntax example

```lisp
(lambda (class . initargs)
      (cond ((or (eq? class <class>)
     (eq? class <entity-class>))
       (let* ((new (%allocate-instance
        class
        (length the-slots-of-a-class)))
        (dsupers (getl initargs 'direct-supers '()))
        (dslots  (map list
          (getl initargs 'direct-slots  '())))
        (cpl     (let loop ((sups dsupers)
          (so-far (list new)))
          (if (null? sups)
              (reverse so-far)
              (loop (class-direct-supers
               (car sups))
              (cons (car sups)
              so-far)))))
        (slots (apply append
          (cons dslots
          (map class-direct-slots
               (cdr cpl)))))
        (nfields 0)
        (field-initializers '())
        (allocator
          (lambda (init)
      (let ((f nfields))
        (set! nfields (+ nfields 1))
        (set! field-initializers
        (cons init field-initializers))
        (list (lambda (o)   (get-field  o f))
        (lambda (o n) (set-field! o f n))))))
        (getters-n-setters
          (map (lambda (s)
           (cons (car s)
           (allocator (lambda () '()))))
         slots)))

         (slot-set! new 'direct-supers      dsupers)
         (slot-set! new 'direct-slots       dslots)
         (slot-set! new 'cpl                cpl)
         (slot-set! new 'slots              slots)
         (slot-set! new 'nfields            nfields)
         (slot-set! new 'field-initializers (reverse
               field-initializers))
         (slot-set! new 'getters-n-setters  getters-n-setters)
         new))
      ((eq? class <generic>)
       (let ((new (%allocate-entity class
            (length (class-slots class)))))
         (slot-set! new 'methods ())
         new))
      ((eq? class <method>)
       (let ((new (%allocate-instance
       class
       (length (class-slots class)))))
         (slot-set! new
        'specializers
        (getl initargs 'specializers))
         (slot-set! new
        'procedure
        (getl initargs 'procedure))
         new)))))
```



