include("../src/read/read_datasets.jl");
include("../src/utils.jl");
include("../src/opq/OPQ.jl");
include("../src/chainq/chainq.jl");
include("../src/lsq/LSQ.jl");
include("../src/linscan/Linscan.jl");
using JLD;

function demo_lsq(
  dataset_name="SIFT1M",
  nread::Integer=Int(1e5)) # Increase this to 1e5 to use the full dataset

  # === Hyperparams ===
  m       = 16 # In LSQ we use m-1 codebooks 23
  h       = 256 #2048
  verbose = true
  nquery  = Int(1e4)
  knn     = Int(1e3) # Compute recall up to
  b       = Int( log2(h) * m )
  niter   = 1
  lsqiter = 1

  # === OPQ initialization ===
  x_train              = read_dataset(dataset_name, nread )
  d, _                 = size( x_train )
  C, B, R, train_error = train_opq(x_train, m, h, niter, "natural", verbose)
  @printf("Error after OPQ is %e\n", train_error[end])
  # #
  # # === ChainQ initialization ===
  # # # B                    = convert( Matrix{UInt16}, B )
  C, B, R, train_error = train_chainq( x_train, m, h, R, B, C, niter )
  @printf("Error after ChainQ is %e\n", train_error[end])
  # @save("/home/zhangqingsong/Revisiting-AQ/a/local-search-quantization-master/demos/ChainQ_data_16_256_new.jld",C, B, R, train_error);
  # @load("/home/zhangqingsong/Revisiting-AQ/a/local-search-quantization-master/demos/ChainQ_data_16_256_lsmr.jld")
  # === LSQ train ===
  ilsiter = 8
  icmiter = 4
  randord = true
  npert   = 4
  # my_B = vector{Matrix{Float32}}(m)
  # my_C = vector{Matrix{Float32}}(m)
  # for i = 1:m:
  #   my_C[i] = C
  #   my_B[i]  = B
  # end
  #R = eye(128, 128)
  #B = zeros(nread, 256)
  #for j = 1:nread:
  #  tmp = sample(1:256,8,replace=false)
  #  for i = tmp:
  #    B[j,i] = 1
  #  end
  #end

  C, B, cbnorms, B_norms, obj = train_lsq( x_train, m, h, R, B, C, lsqiter, ilsiter, icmiter, randord, npert )
  cbnorms = vec( cbnorms[:] )
  # #
  # #
  # @save("/home/zhangqingsong/Revisiting-AQ/a/local-search-quantization-master/demos/all_Train_data_16_256_new.jld",C, B, cbnorms, B_norms, obj);
  # @load("/home/zhangqingsong/Revisiting-AQ/a/local-search-quantization-master/demos/all_Train_data_16_256_lsmr.jld")
  # === Encode the base set ===
  nread_base   = Int(1e6)
  x_base       = read_dataset(dataset_name * "_base", nread_base )
  B_base       = randinit(nread_base, m, h) # initialize B at random

  ilsiter_base = 1 # LSQ-16 in the paper
  @time for i = 1:ilsiter_base
    @printf("Iteration %02d / %02d\n", i, ilsiter_base)
    @time B_base = encoding_icm( x_base, B_base, C, icmiter, randord, npert, verbose )
    @everywhere gc()
  end
  base_error = qerror( x_base, B_base, C )
  @printf("Error in base is %e\n", base_error)

  # Compute and quantize the database norms
  B_base_norms = quantize_norms( B_base, C, cbnorms )
  db_norms     = vec( cbnorms[ B_base_norms ] )

  # === Compute recall ===
  x_query = read_dataset( dataset_name * "_query", nquery, verbose )
  gt      = read_dataset( dataset_name * "_groundtruth", nquery, verbose )
  if dataset_name == "SIFT1M" || dataset_name == "GIST1M"
    gt = gt + 1;
  end
  gt           = convert( Vector{UInt32}, gt[1,1:nquery] )
  B_base       = convert( Matrix{UInt16}, B_base-1 )
  B_base_norms = convert( Vector{UInt16}, B_base_norms-1 )
  # @save("/home/zhangqingsong/Revisiting-AQ/local-search-quantization-master/demos/all_search_data_8_11_11.jld",B_base, x_query, C, db_norms, knn, d, gt);
  # @load("/home/zhangqingsong/Revisiting-AQ/local-search-quantization-master/demos/alldata.jld")
  print("Querying m=$m ...h=$h... ")
  @time dists, idx = linscan_lsq( B_base, x_query, C, db_norms, eye(Float32, d), knn )
  println("done")

  idx = convert( Matrix{UInt32}, idx );
  rec = eval_recall( gt, idx, knn )

end

# train
@time demo_lsq()
