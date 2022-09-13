# Description and explanations of the methods used in OAT

## Index
1. [Search of the centroid](#Centroid)  
	a. [The "mean" way](#Mean)  
	b. [The "gradient" method](#Gradient)  
	c. [The randomized method](#Rng)  
2. [Clustering to identify the organoïd](#Clustering)  
3. [Drift and volume](#DriftandVolume)  
	a. [Drift](#Drift)  
	b. [Volume](#Volume)
4. [Rotation axis](#Rotaxis)  
5. [Angular velocity](#Angle)  

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

## 2. Clustering to identify the organoïd <a name="Clustering"></a>
As specified in the 1.b part, sometimes, the organoïd is not well segmented for some reason. One of them can be the proximity with another organoïd for example.

Clustering spots is one of the solution to differientiate cells that belong to the organoïd from cells that don't. But, this is not a easy and simple problem as there are several clustering algorithms to choose from, they come with they're set of paramaters and those are generally not identicall between samples. That means that there is a lot of trial and errors befor getting a well clustered organoïd.  

That is for those reasons that it is segmenting is preferable to clustering.

OAT comes with an implementation of DBSCAN which is by test, the most reliable clustering method for this kind of problems.
<insert DBSCAN algorithm explanations>

So, the basic clustering of OAT simply run a DBSCAN on the spots coordinates using some predetermined values but, as said before, they will surely not work on other dataset.

Once the spots have been clustered, the results are in the form of a number designating a cluster ID (1 for example), associated with each spots.  

The cluster that is then considered to be the organoïd is the one that has the more points. This choice is based on the assumption that the organoïd contains more cells than there are "bad" cells. This work since the organoïd have been more or less well segmented and the majority of bad cells have been removed.  

The final clustering results are saved in boolean : True for the spots that belongs to the organoïd and False otherwise.  

There is an extra step to reinforce the results that is completely optional.
In OAT, it is called clustering on distances. Prior to the DBSCAN clustering, another DBSCAN is runned on the distances between the centroïd and the spots. The centroïd is found using the methods above.  
The idea is that the centroïd will leans toward the center of the organoïd. Given that the organoïd is spherical, spots belonging to it will be in the same range distance-wise. That should lead to 2 or more clear groups on an histogram, one being the organoïd, others being "bad" cells clusters.  
<Add histogram showing the clustering>

The clustering results or ID are summed with the results of the second clustering to get a more refined clustering. The selection process is then runned on the assembled clusters ID.

## 3. Drift and volume <a name="DriftandVolume"></a>
### a. Drift <a name="Drift"></a>
The drift represent the displacement of the whole organoïd between time points. In order to compute it, the centroïd is computed for each time point, then the displacement vectors are extracted from the displacement between 2 time centroïds.  
Once again, the methods to compute the centroïd are available in the first part.  
The algebric distance is then recovered with the magnitude of the vectors.  

### b. Volume <a name="Volume"></a>
To compute the volume of the organoïd at any time points, the convex hull algorithm from Scipy is runned on the spots. This return the volume of the object.
<insert animation on how the convex hull algorithm works>

## 4. Rotation axis <a name="Rotaxis"></a>
The axis of rotation of the organoïd helps determining the angular velocity of each cells. To compute it, a PCA is runned on the displacement vectors. The PCA (Principal Component Analysis) allows to reduce the dimension of a multiple dimension problem to a 2 dimension problem. The displacement vectors are used because a PCA on those gives the plane which describe most of the problem.  
<Show a basic rotation>
On this simple example it is easy to find the plane that decribe most of the problem. That plane is the XY plane because when represented on this plane, diplacement vectors clearly show the main rotation happening to the sphere.  
<Add view for each plane>  

The PCA give 2 vectors corresponding to the unitary vectors for each axis of the PCA plane. Knowing that the rotation axis is perpendicular to that plane, one of its directory vectors is the crossproduct of the 2 unitary vectors from the PCA plane.  
<Adding visualization>

In OAT, only this directory vector is saved and refered as the rotation axis.  

## 5. Angular velocity <a name="Angle"></a>
The angular velocity of each cell is computed using the following formula :

magnitude( crossproduct(r, displacement)/r^(2) )  
<Add real math formulas>

where r = dotproduct(coord, V1)*V1+dotproduct(coord, V2)
and displacement is the displacement vector.

coord are the coordinates of the spot, V1 and V2 are the PCA plane vectors.