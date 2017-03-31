public class Solution {

    public static void main(String[] args) {
        // Prints "Hello, World" to the terminal window.
	String a = "au";
	
	System.out.println(lengthOfLongestSubstring(a));
    }
    
    public static int lengthOfLongestSubstring(String s) {
	if (s.length() == 0){
	    return 0;
	}
	
	int bestLength = 1;
	int currentStart = 0;
	int currentEnd = 1;
	int currentLength = 1;
	while(currentEnd < s.length()){
	   
	    char c = s.charAt(currentEnd);
	    
	    int pos = s.substring(currentStart, currentEnd).indexOf(c);
	   
	    if (pos >= 0){
		currentLength = currentEnd - currentStart;
		bestLength = currentLength > bestLength ? currentLength: bestLength;
		currentStart = currentStart + pos + 1;
	    }
	    currentEnd += 1;
	}
	
	currentLength = currentEnd - currentStart;
	bestLength = currentLength > bestLength ? currentLength: bestLength;
	return bestLength;
	    
    }
}


