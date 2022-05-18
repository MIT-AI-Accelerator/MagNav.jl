using Coverage

# coverage summary, provided as % covered
cd(joinpath(@__DIR__,"../..")) do
    Codecov.submit_local(process_folder())
    (cl,tl) = get_summary(process_folder())
    println("($(cl/tl*100)%) covered")
end
