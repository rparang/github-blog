Title: Foobar with Google: Peculiar Balance
Date: 2014-07-31
Slug: foobar-with-google-peculiar-balance


*Originally published in July 2014*

There has been [recent](https://news.ycombinator.com/item?id=8588080) [buzz](https://www.businessinsider.com/google-hiring-developers-foobar-challenge-2014-11) about a Google recruiting method called [Foobar](https://foobar.withgoogle.com/). Supposedly if you search enough programming-related issues like "python mutex lock," your account is flagged and you're offered to play a game with a series of programming challenges. If you do well, people claim you're invited for an interview.

Well, I wasn't offered to play this game. Thankfully, my friend was, and he shared one of the challenges below:

> ### Peculiar balance  
>
> Can we save them? Beta Rabbit is trying to break into a lab that contains the only known zombie cure - but there's an obstacle. The door will only open if a challenge is solved correctly. The future of the zombified rabbit population is at stake, so Beta reads the challenge:
>
> There is a scale with an object on the left-hand side, whose mass is given in some number of units. Predictably, the task is to balance the two sides. But there is a catch: You only have this peculiar weight set, having masses 1, 3, 9, 27, ... units. That is, one for each power of 3. Being a brilliant mathematician, Beta Rabbit quickly discovers that any number of units of mass can be balanced exactly using this set.
>
> To help Beta get into the room, write a method called `answer(x)`, which outputs a list of strings representing where the weights should be placed, in order for the two sides to be balanced, assuming that weight on the left has mass x units.
>
> The first element of the output list should correspond to the 1-unit weight, the second element to the 3-unit weight, and so on. Each string is one of:
>
> - `"L"` : put weight on left-hand side  
> - `"R"` : put weight on right-hand side  
> - `"-"` : do not use weight  
>
> To ensure that the output is the smallest possible, the last element of the list must not be `"-"`.
>
> `x` will always be a positive integer, no larger than 1,000,000,000.
>
> #### Test cases  
> Inputs: `(int) x = 2`  
> Output: `(string list) ["L", "R"]`  
>
> Inputs: `(int) x = 8`  
> Output: `(string list) ["L", "-", "R"]`

The idea that any mass on the scale can be balanced using weights with only powers of three (with only one weight per power of 3) doesn't seem intuitive.

It turns out this challenge has its roots in ternary, a base-3 numeral system similar to binary. And there is a way of representing ternary in a balanced form, where each digit is represented with either "+", "0", or "-" (or something similar). Using this system, *any* number can be represented. I find this fact fascinating.

For example, the number 23 in the decimal system in ternary is 212 (`2*3^0 + 1*3^1 + 2*3^2`). We can represent 212 in a balanced form as `+0--` the following way: moving from right to left, convert each 2 value to a "-+", where the + takes the 2's position and the negative character is carried to the next column. The negative carried value is added to the existing column and the process continues. There are [nice write-ups](https://en.wikipedia.org/wiki/Balanced_ternary#Conversion_from_ternary) on conversion to ternary.

This method seems ideal for our challenge.

First, we'll convert our initial scale value to ternary with a `decimal_to_ternary(num)` method. Then, we'll balance the ternary result using `balance(num)` based on our strategy above, taking into account that we'll represent our "+", "0", and "-" characters as either `"L"`, `"R"`, and `"-"`.

A JavaScript method for converting to ternary with a few comments is below:

```javascript
function decimal_to_ternary(num) {
  
  if (num == 0) return 0;

  var i = 0;
  var result = [];

  // Looking for the largest power of 3 where 3^i
  // is not larger than num. Add 0's for the index 
  // in the meantime
  while(num > 0) {
    while (num / Math.pow(3, i) >= 1) {
      result[i] = 0;
      i++;
    }

    // We leave our loop with i being 1 too large
    i--;

    // Since we can use up to 2, see if we can fit 2
    if (2 * Math.pow(3, i) <= num) {
      result[i] = 2; 
      num = num - 2 * Math.pow(3, i);
    } else {
      result[i] = 1;
      num = num - Math.pow(3, i);
    }
    i = 0;
  }
  return parseInt(result.reverse().join(''));
}
```

Next, we can take the result and balance it with our challenge's notation:

```javascript
function balance(num) {

  var carry = 0;
  var i, temp;

  // Need to turn our number into an array. Reversing it since
  // we're working right to left through the conversion
  var array = num.toString().split('').reverse().map(function(a,b) {
    return parseInt(a)
  });

  for (i = 0; i < array.length; i++) {
    temp = array[i] + carry;
    carry = 0;
    switch(temp) {
      case 3:
        array[i] = "-";
        carry = 1;
        break;
      case 2:
        array[i] = "L"
        carry = 1;
        break;
      case 1:
        array[i] = "R";
        break;
      default:
        array[i] = "-";
        break;
    }
  };
  if (carry == 1) {
    array.splice(array.length, 0, "R")
  }
  return array;
}
```

With these two, we can solve our problem:

```javascript
function answer(num) {
  return balance(decimal_to_ternary(num));
}
```

We can test this by running `answer(546)` for the scenario where our rabbit super hero sees an initial weight of 546 on the left side of the scale. Our output list becomes `["-", "L", "R", "L", "R", "L", "R"]`, where our left side is `546 + 3^1 + 3^3 + 3^5` and our right side is `3^2 + 3^4 + 3^6`. Both sides become balanced at a weight of 819.

As a side note: This is likely not an efficient way to solve this problem. It's possible to convert a decimal number directly to balanced ternary, but breaking the steps up helped me better learn the process.