Title: Coderbyte Challenge: Bracket Matcher
Date: 2014-07-31
Tags: javascript, interviewing, google, blog
Slug: coderbyte-challenge-bracket-matcher


*almost 7 years ago*
Recently I've been attempting the programming challenges on [Coderbyte](https://web.archive.org/web/20210622222916/http://coderbyte.com/CodingArea/Challenges/). Coderbyte is a site that has a handful of questions that allots points given how 1) fast you complete the challenge and 2) how correct your answer is.

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

Let's see how these perform against one another. To do this, I need some text with lots of parentheses. Some [Lisp sample code](https://web.archive.org/web/20210622222916/http://rezaparang.com/lisp.txt) should work just fine.


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



