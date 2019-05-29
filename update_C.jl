#update_C.jl
# my_C = update_C( X,  B, my_C);
using HDF5, Clustering


function update_C{T <: AbstractFloat}(
  X::Matrix{T},           # d-by-n 
  B::Matrix{Int16},       # m-by-n
  my_C::Vector{Matrix{Float32}}       # m-d-k
)
  d, n   = size(X);
  m, _   = size(B);
  B_vector = zeros(n,256);

  for i=1:n
    for j = 1:m
        B_vector[i, B[j,i]]=1;
    end
  end

  print("starting opt_C");
  delta=0.01*B_vector'*(B_vector*(my_C[1])'-X');
  temp=(my_C[1])'-delta;
  print("finished opt_C");
  for i=1:m
    my_C[i]=temp';
  end 
  return my_C


#  my_C[1]=X*B_vector*inv((B_vector'*B_vector+0.000001*eye(256,256)));
#  for i=1:m
#    my_C[i]=my_C[1];
#  end 
#  return my_C
end
