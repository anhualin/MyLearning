public class Solution {

    public static void main(String[] args) {
        // Prints "Hello, World" to the terminal window.
	int[] nums1 = { 
		    100, 200, 300,
		    400, 500, 600, 
		    700, 800, 900, 1000
	};
	int[] nums2 = { 
		    1, 200, 300,
		    400, 500, 600, 
		    700, 800, 900
	};
	
	System.out.println(findMedianSortedArrays(nums1, nums2));
    }
    
    public static double findMedianSortedArrays(int[] nums1, int[] nums2) {
	int len1 = nums1.length;
	int len2 = nums2.length;
	
	if (len1 == 0){
	    if (len2 % 2 == 0){
		return (double)(nums2[len2/2 - 1] + nums2[len2/2])/2.0;
	    }else{
		return (double)(nums2[len2/2]);
	    }
	}
	if (len2 == 0){
	    if (len1 % 2 == 0){
		return (double)(nums1[len1/2 - 1] + nums1[len1/2])/2.0;
	    }else{
		return (double)(nums1[len1/2]);
	    }
	}
	int half = (len1 + len2)/2;
	int l1 = 0;
	int u1 = len1 - 1;
	int l2 = 0;
	int u2 = len2 - 1;
	int x = (l1 + u1)/2;
        int y = (l2 + u2)/2;

	while(u1 - l1 >= 2 && u2 - l2 >= 2){
	   
	    if (nums1[x] >= nums2[y]){
		if (x + y >= half - 1){
		    u1 = x;
		    x = (l1 + u1)/2;
		}else{
		    l2 = y;
		    y = (l2 + u2)/2;
		}
	    }else{
		if (x + y >= half - 1){
		    u2 = y;
		    y = (l2 + u2)/2;
		}else{
		    l1 = x;
		    x = (l1 + u1)/2;
		}
	    }
		
	   }
	if (u1 - l1 >= 2){
	    int [] tmp = nums1;
	    nums1 = nums2;
	    nums2 = tmp;
	    int t = l1;
	    l1 = l2;
	    l2 = t;
	    t = u1;
	    u1 = u2;
	    u2 = t;
	    len1 = nums1.length;
	    len2 = nums2.length;
	}
	while(u2 - l2 >=2){
	    y = (l2 + u2)/2;
	    if (nums2[y] < nums1[l1]){
		if (nums1[l1] > nums1[0] || y < half){
		    l2 = y;
		}else{
		    if ((len1 + len2) % 2 == 0){
			return (double)(nums2[half - 1] + nums2[half])/2.0;
		    }else{
			return (double)(nums2[half]);
		    }
		}
	    }else if (nums2[y] > nums1[u1]){
		if (nums1[u1] < nums1[len1 - 1] || len2 - y - 1 < half){
		    u2 = y;
		}else{
		    if ((len1 + len2) % 2 == 0){
			return (double)(nums2[len2 - half - 1] + nums2[len2 - half])/2.0;
		    }else{
			return (double)(nums2[len2 - half - 1]);
		    }
		}
	    }else if (nums2[y] == nums1[l1]){
		if (l1 + y < half - 1){
		    l2 = y;
		}else{
		    u2 = y;
		}
	    }else if (nums2[y] == nums1[u1]){
		if (u1 + y < half - 1){
		    l2 = y;
		}else{
		    u2 = y;
		}
	    }else{
		if ((l1 + y + 1) >= half){
		    u2 = y;
		}else{
		    l2 = y;
		}
	    }
	       
	
		    
	}
	System.out.println(nums1[l1]);
	System.out.println(nums1[u1]);
	System.out.println(nums2[l2]);
	System.out.println(nums2[u2]);
	return (double)(len1 + len2);
    }
    
   
}


