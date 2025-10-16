
// circleInBoxConservative --
//
// Tests whether circle with center (circleX, circleY) and radius
// `circleRadius` *may intersect* the box defined by coordinates for
// it's left and right sides, and top and bottom edges.  For
// efficiency, this is a conservative test.  If it returns 0, then the
// circle definitely does not intersect the box.  However a result of
// 1 does not imply an intersection actually exists.  Further tests
// are needed to determine if an intersection actually exists.  For
// example, you could continue with actual point in circle tests, or
// make a subsequent call to circleInBox().
// Note: For a valid Box, you will want to use boxR >= boxL and 
// boxT >= boxB. 
__device__ __inline__ int
circleInBoxConservative(
    // int blockId,
    float circleX, float circleY, float circleRadius,
    float boxL, float boxR, float boxT, float boxB)
{

    // expand box by circle radius.  Test if circle center is in the
    // expanded box.

    if ( circleX >= (boxL - circleRadius) &&
         circleX <= (boxR + circleRadius) &&
         circleY >= (boxB - circleRadius) &&
         circleY <= (boxT + circleRadius) ) {
        // printf("in conservative, blockId = %d\n", blockId);
        // printf("boxL - circleRadius = %f, boxR + circleRadius = %f, boxB - circleRadius = %f, boxT + circleRadius = %f\n", 
        //     boxL - circleRadius, boxR + circleRadius, boxB - circleRadius, boxT + circleRadius);
        return 1;
    } else {
        return 0;
    }
}


// circleInBox --
//
// This is a true circle in box test.  It is more expensive than the
// function circleInBoxConservative above, but it's 1/0 result is a
// definitive result.
// Note: For a valid Box, you will want to use boxR >= boxL and 
// boxT >= boxB. 
__device__ __inline__ int
circleInBox(
    // int blockId,
    float circleX, float circleY, float circleRadius,
    float boxL, float boxR, float boxT, float boxB)
{

    // clamp circle center to box (finds the closest point on the box)
    float closestX = (circleX > boxL) ? ((circleX < boxR) ? circleX : boxR) : boxL;
    float closestY = (circleY > boxB) ? ((circleY < boxT) ? circleY : boxT) : boxB;

    // is circle radius less than the distance to the closest point on
    // the box?
    float distX = closestX - circleX;
    float distY = closestY - circleY;

    if ( ((distX*distX) + (distY*distY)) <= (circleRadius*circleRadius) ) {
        // if(0 == blockId) {
        //     printf("blockId = %d\n", blockId);
        //     printf("circleInBox: circleX = %f, circleY = %f, circleRadius = %f, boxL = %f, boxR = %f, boxT = %f, boxB = %f\n", 
        //         circleX, circleY, circleRadius, boxL, boxR, boxT, boxB);
        //     printf("circleInBox: closestX = %f, closestY = %f\n", closestX, closestY);
        //     printf("circleInBox: distX = %f, distY = %f\n", distX, distY);
        //     printf("circleInBox: (distX*distX) + (distY*distY) = %f, (circleRadius*circleRadius) = %f\n", 
        //         (distX*distX) + (distY*distY), (circleRadius*circleRadius));
        // }
        return 1;
    } else {
        return 0;
    }
}

