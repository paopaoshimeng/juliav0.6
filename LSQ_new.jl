
# Full-dimensional local search quantization
using HDF5, Clustering

include("../utils.jl");
include("../initializations.jl");
include("../update_C.jl");
include("../encodings/encode_icm.jl");

function train_lsq{T <: AbstractFloat}(
  X::Matrix{T},         # d-by-n matrix of data points to train on.
  m::Integer,           # number of codebooks
  h::Integer,           # number of entries per codebook
  R::Matrix{T},         # init rotation
  B::Matrix{Int16},     # init codes
  C::Matrix{T}, # init codebooks
  niter::Integer,       # number of optimization iterations
  ilsiter::Integer,     # number of ILS iterations to use during encoding
  icmiter::Integer,     # number of iterations in local search
  randord::Bool,        # whether to use random order
  npert::Integer,       # The number of codes to perturb
  V::Bool=false)        # whether to print progress
  m=8

  println("**********************************************************************************************");
  println("Doing local search with $m codebooks, $npert perturbations, $icmiter icm iterations and random order = $randord");
  println("**********************************************************************************************");

  my_C = Vector{Matrix{Float32}}(m)
  d, n = size( X );
  RX = R' * X;
  for i = 1:m
    my_C[i] = C
  end
  @printf("%3d %e \n", -2, qerror( X, B, my_C ));

  for i = 1:ilsiter
    B = encoding_icm( X, B, my_C, icmiter, randord, npert, V );
    @everywhere gc()
  end
  @printf("%3d %e \n", -1, qerror( X, B, my_C ));

  my_C = update_C(X,B,my_C);
  obj = zeros( Float32, niter );
  @printf("%3d %e \n", 0, qerror( X, B, my_C ));

  for iter = 1:niter

    for i = 1:ilsiter
      B = encoding_icm(X,B,my_C,icmiter,randord,npert,V);
      @everywhere gc()
    end    
    # Update the codebooks
    my_C=update_C(X,B,my_C);
    obj[iter] = qerror(X,B,my_C);

    @printf("%3d %e \n", iter, obj[iter]);
  end


  C=my_C

  # Get the codebook for norms
  CB = reconstruct(B, my_C);
  dbnorms = zeros(Float32, 1, n);
  for i = 1:n
     for j = 1:d
        dbnorms[i] += CB[j,i].^2;
     end
  end

  # Quantize the norms with plain-old k-means
  dbnormsq = kmeans(dbnorms, h);
  cbnorms  = dbnormsq.centers;

  # Add the dbnorms to the codes
  B_norms  = reshape( dbnormsq.assignments, 1, n )

  return my_C, B, cbnorms, B_norms, obj

end
