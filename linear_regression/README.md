# Multivariate linear regression

![gradient descent](animation.gif)

The gradient descent equation used for implementation:

![gradient descent](https://latex.codecogs.com/svg.latex?%5Ctheta_%7Bj%7D%20%3A%3D%20%5Ctheta_%7Bj%7D%20-%20%5Calpha%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29-y%5E%7B%28i%29%7D%29%5Ccdot%20x_%7Bj%7D%5E%7B%28i%29%7D)

where:

 - `θj` - parameters vector which should be find!! Why we actually here 😃
 - `x` - our features vector
 - `y` - actual result values (e.g. labels)
 - `α` - learning step
 - `hθ` - hypothesis function or:
  
     - ![](https://latex.codecogs.com/svg.latex?h_%7B%5Ctheta%7D%28x%29%20%3D%20%5Ctheta_%7B0%7D%20&plus;%20%5Ctheta_%7B1%7Dx_%7B1%7D%20&plus;%20%5Ctheta_%7B2%7Dx_%7B2%7D%20&plus;%20...%20&plus;%20%5Ctheta_%7Bn%7Dx_%7Bn%7D)

With Python and Numpy it can be written this way:
```python
theta = theta - alpha * 1 / m * X.T @ (X @ theta - y)
```
where:
- `@` - matrix multiplication operation
- `.T` - property returning the transposed matrix