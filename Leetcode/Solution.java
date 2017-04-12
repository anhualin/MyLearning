import java.util.*;
public class Solution {
    
    public static void main(String[] args) {
       
	Random ran = new Random();
	int N1 = ran.nextInt(100) + 1;
	int N2 = ran.nextInt(100) + 1;
	int[] a3 = new int[N1 + N2];
	int[] a1 = new int[N1];
	for(int i = 0; i < N1; i++){
	    a1[i] = ran.nextInt(100);
	    a3[i] = a1[i];
	}
	Arrays.sort(a1);
	int[] a2 = new int[N2];
	for(int i = 0; i < N2; i++){
	    a2[i] = ran.nextInt(100);
	    a3[N1 + i] = a2[i];
	}
	Arrays.sort(a2);
	Arrays.sort(a3);
	double result = findMedianSortedArrays(a1, a2);
	double r;
	if ((N1 + N2)%2 == 0){
	    r = (double)(a3[(N1 + N2)/2 - 1] + a3[(N1 + N2)/2])/2.0;
	}else{
	    r = (double)(a3[(N1 + N2)/2]);
	}
	System.out.println(N1);
	System.out.println(N2);
	System.out.println(result -r);
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

	int gotten = 0;
	int total = 0;
	int [] candidates = {nums1[l1], nums1[u1], nums2[l2], nums2[u2]};
	if ((len1 + len2) % 2 == 0){
	    
	    for(int e: candidates){
		Positions pos = findPos(e, nums1, nums2);
		int pos0 = pos.getLeft();
		int pos1 = pos.getRight();
		if (pos0 <= half && pos1 >= half + 1){
		    return (double)e;
		}
		if ((pos0 <= half && pos1 >= half) ||
		    (pos0 <= half + 1 && pos1 >= half + 1)){
		    if (gotten == 0 || total != e){
			gotten ++;
			total += e;
			if (gotten == 2){
			return (double)(total)/2.0;
			}
		    }
		}
	    }
	}else{
	    for(int e: candidates){
		Positions pos = findPos(e, nums1, nums2);
		int pos0 = pos.getLeft();
		int pos1 = pos.getRight();
		if (pos0 <= half + 1 && pos1 >= half + 1){
		    return (double) e;
		}
	    }
	}
	
	
	return (double)(len1 + len2);
    }
    
    public static Positions findPos(int x, int[] nums1, int[] nums2){
	int pl1 = findLower(x, nums1);
	int pl2 = findLower(x, nums2);
	int pu1 = findUpper(x, nums1);
	int pu2 = findUpper(x, nums2);
	int left = pl1 + pl2 + 3;
	int right = pu1 + pu2;
	return new Positions(left, right);
	
    }
    public static int findLower(int x, int[] nums){
	int N = nums.length;
	int pl, pu;
	if (x <= nums[0]){
	    return -1;
	}else if (x > nums[N - 1]){
	    return N - 1;
	}else{
	    pl = 0;
	    pu = N - 1;
	    while(pu - pl > 1){
		int z = (pu + pl)/2;
		if (x > nums[z]){
		    pl = z;
		}else{
		    pu = z;
		}
	    }
	}
	return pl;
    }
    public static int findUpper(int x, int[] nums){
	int N = nums.length;
	int pl, pu;
	if (x < nums[0]){
	    return 0;
	}else if (x >= nums[N - 1]){
	    return N;
	}else{
	    pl = 0;
	    pu = N - 1;
	    while(pu - pl > 1){
		int z = (pu + pl)/2;
		if (x < nums[z]){
		    pu = z;
		}else{
		    pl = z;
		}
	    }
	}
	return pu;
    }
}


  class Positions{
     private int left;
     private int right;
     Positions(int left, int right){
	 this.left = left;
	 this.right = right;
     }
     int getLeft(){
	 return left;
     }
     int getRight(){
	 return right;
     }
  }
	
