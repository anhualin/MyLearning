import java.util.*;
public class Solution {
    
    public static void main(String[] args) {
	String a = longestPalindrome("a");
  
	System.out.println(longestPalindrome("a"));
	System.out.println(longestPalindrome("abccbax1234321xae"));
	System.out.println(longestPalindrome("a123bxyyxbaa"));
	System.out.println(longestPalindrome("abbbbbb"));
    }

    public static String longestPalindrome(String s) {
	int LS = s.length();
	if (LS <= 1){
	    return s;
	}

	int bestCenter = 0;
	int bestLength = (s.charAt(0) == s.charAt(1)) ? 2 : 1;
	int curr = 1;



	String sReverse;
	int half;
	while(curr + bestLength / 2 < LS){
	    half = (bestLength % 2 == 0)? bestLength / 2 : bestLength / 2 + 1;
	 
	    /* check odd palindrome centered at curr */
	    if ((curr - half >= 0) && (curr + half < LS)){
		sReverse = new StringBuilder(s.substring(curr + 1, curr + half + 1)).reverse().toString();
		if (s.substring(curr - half, curr).equals(sReverse)){
		    half++;
		    while((curr - half >=0) && (curr + half < LS) &&
			  (s.charAt(curr - half) == s.charAt(curr + half)))
			half++;
		    half--;
		    bestLength = 2 * half + 1;
		    bestCenter = curr;
		}
	    }

	    /*  check even palindrome centered at curr */
	    half = bestLength / 2 + 1;
	    if ((curr - half + 1 >= 0) && (curr + half < LS)){
		sReverse = new StringBuilder(s.substring(curr + 1, curr + half + 1)).reverse().toString();
		if (s.substring(curr - half + 1, curr + 1).equals(sReverse)){
		    half++;
		    while((curr - half + 1 >=0) && (curr + half < LS) &&
			  (s.charAt(curr - half + 1) == s.charAt(curr + half)))
			half++;
		    half--;
		    bestLength = 2 * half;
		    bestCenter = curr;
		}
	    }
	    curr++;
	}
	half = bestLength / 2;
	String palindrome = (bestLength %2 == 0)? s.substring(bestCenter - half + 1, bestCenter + half + 1) :
	    s.substring(bestCenter - half, bestCenter + half + 1);
	return palindrome;
		       
    }
  
}
	
