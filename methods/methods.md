# Description and explanations of the methods used in OAT

## Index
1. [Search of the centroid](#Centroid)  
	a. [The "mean" way](#Mean)  
	b. [The "gradient" method](#Gradient)  
	c. [The randomized method](#Rng)  
2. [Clustering to identify the organoïd](#Clustering)  
3. [Drift and volume](#DriftandVolume)  
4. [Rotation axis](#Rotaxis)  
5. [Angular velocity](#Angvel)  

## Glossary
- spot(s) : point that represent a cell's center. Their coordinates can be written as (xi, yi, zi) with i being the ID of the spot.

## 1. Search of the centroid <a name="Centroid"></a>
The centroid is here defined as the point located in the center of the organoïd, equally distant with all spots forming the organoïd.  
We will see 3 methods to find it, available within OAT.  

### a. The "mean" way <a name="Mean"></a>
This is the simplest method and the most accurate when the organoïd is well segmented.  

To compute it, the mean of the coordinates on each axis is computed.
<insert some math here>

It can be difficult and time consuming to segment an organoïd through time without keeping a part of another one or "bad" cells. In those cases, the mean method is not able to give the correct centroïd.
<insert examples>

### b. The "gradient" method <a name="Gradient"></a>
This method suppose that there are more cells being part of the organoïd and that those ones are separated enough from the "bad" cells.

One another way to see the centroïd could be the point that minimizes the distance with every spot. Or also, the point that minimizes the sum of the distance with every spot. 

Using the second statement, computing every sum for every point in the space is not efficient because for a simple 40x40x40 cube pixel volume, the number of sums to compute is 64.000, let aside the number of spots in the space.  

To circumvent this limitation, OAT uses a well known method which relies on the gradient slope to converge to a minima or maxima depending on the use.

The algorithm starts by taking a random point in space and computes the variation of the sum of distances over a small distance for all axes, one at a time. 
<insert basic definition of the derivative>
Math says that the smaller the distance is, the more the value of that fraction tends to be the value of the derivative at the point coordinates.

So, to get more math friendly terms, we compute the partial derivative of the sum of the distances on each axis. Then, the next point is selected using the slope of the gradient.

As a reminder, the partial derivative gives the slope of the sum function on the given axis, at the coordinates of the point. A positive slope indicates that the function increases as the value on that given axis increases. As we search for the minima, the algorithm will go with the descending slope.

The next point is then selected using a user customizable variable called speed for speed of research. The next point coordinates are given by the equations :  
X(i+1) = X(i)+speed*(-sign(dF/dX))  
Y(i+1) = Y(i)+speed*(-sign(dF/dY))  
Z(i+1) = Z(i)+speed*(-sign(dF/dZ))  

In this case, we take the invert of the sign of the partial derivative because OAT searches for the minima. If the sign is positive, the function increases as the value rises, so we want to go in the opposite direction and decrease the value.

Also, the higher the "speed", the fastest the algorithm will converge to the minima. But, if the speed is too high, the algorithm will not converge due to the lack of precision implied by larger spacing between points. 

The above steps are repeated until the function variation is very small.

### c. The "randomizing" way <a name="Rng"></a>
The name of this method is specific to OAT.

Another way to circumvent the bad results of the "mean" method than the gradient method is to take multiple subset of spots, compute the "mean" centroïd and take the mean of the "mean" centroids afterwards.

This method suppose that the "bad" cells are very few.

In theory, if multiple "mean" centroïd are computed on 10, randomly chosen, spots; the mean of those centroïd will be closer to the organoïd's centroïd.