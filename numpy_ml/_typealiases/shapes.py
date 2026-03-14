# Shape excludes Shape0D and scalars not wrapped in arrays
type Shape = tuple[int, ...]

# Shape0D excludes scalars not wrapped in arrays
type Shape0D = tuple[()]

type Shape1D = tuple[int,]
type Shape2D = tuple[int, int]
type Shape3D = tuple[int, int, int]
type Shape4D = tuple[int, int, int, int]
type Shape5D = tuple[int, int, int, int, int]
