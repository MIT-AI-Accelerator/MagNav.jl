"""
    xyz2h5(xyz_xyz::String, xyz_h5::String, flight::Symbol;
           lines::Vector        = [()],
           lines_type::Symbol   = :exclude,
           tt_sort::Bool        = true,
           downsample_160::Bool = true,
           return_data::Bool    = false)

Convert SGL flight data file from .xyz to HDF5.
- Valid for SGL flights:
    - `:Flt1001`
    - `:Flt1002`
    - `:Flt1003`
    - `:Flt1004_1005`
    - `:Flt1004`
    - `:Flt1005`
    - `:Flt1006`
    - `:Flt1007`
    - `:Flt1008`
    - `:Flt1009`
    - `:Flt1001_160Hz`
    - `:Flt1002_160Hz`
    - `:Flt2001_2017`

May take 1+ hr for 1+ GB files. For reference, a 1.23 GB file took 46.8 min to
process using a 64 GB MacBook Pro.

**Arguments:**
- `xyz_xyz`:        path/name of flight data .xyz file (`.xyz` extension optional)
- `xyz_h5`:         path/name of flight data HDF5 file to save (`.h5` extension optional)
- `flight`:         flight name (e.g., `:Flt1001`)
- `lines`:          (optional) selected line number(s) to ONLY include or exclude, must be a vector of 3-element (`line`,`start_time`,`end_time`) tuple(s)
- `lines_type`:     (optional) whether to ONLY `:include` (i.e., to generate testing data) or `:exclude` (i.e., to generate training data) `lines`
- `tt_sort`:        (optional) if true, sort data by time (instead of line)
- `downsample_160`: (optional) if true, downsample 160 Hz data to 10 Hz (only for 160 Hz data files)
- `return_data`:    (optional) if true, return `data` instead of creating `xyz_h5`

**Returns:**
- `data`: if `return_data = true`, internal data matrix
"""
function xyz2h5(xyz_xyz::String, xyz_h5::String, flight::Symbol;
                lines::Vector        = [()],
                lines_type::Symbol   = :exclude,
                tt_sort::Bool        = true,
                downsample_160::Bool = true,
                return_data::Bool    = false)

    xyz_xyz = add_extension(xyz_xyz,".xyz")
    xyz_h5  = add_extension(xyz_h5 ,".h5")

    fields   = xyz_fields(flight) # vector of data field names
    Nf       = length(fields)     # number of data fields (columns)
    ind_tt   = findfirst(fields .== :tt)   # time index (column)
    ind_line = findfirst(fields .== :line) # line index (column)

    # find valid data rows (correct number of columns)
    ind = [(length(split(line)) == Nf) for line in eachline(xyz_xyz)]

    # if 160 Hz data, find valid 10 Hz data rows (tt is multiple of 0.1)
    # probably better ways to do this, but it works ok
    if downsample_160 & (flight in [:Flt1001_160Hz,:Flt1002_160Hz])
        for (i,line) in enumerate(eachline(xyz_xyz))
            ind[i] && (ind[i] = (par(split(line)[ind_tt])+1e-6) % 0.1 < 1e-3)
        end
    end

    N    = sum(ind) # number of valid data rows
    data = zeros(Float64,N,Nf) # initialize data matrix

    @info("reading in file: $xyz_xyz")

    # go through valid data rows and extract data
    for (i,line) in enumerate(eachline(xyz_xyz))
        if ind[i]
            j = cumsum(ind[1:i])[i]
            data[j,:] = par.(split(line))
        end
    end

    # check for duplicated data
    N_tt   = length(unique(data[:,ind_tt  ]))
    N_line = length(unique(data[:,ind_line]))
    N > N_tt + N_line && @info("xyz file may contain duplicated data")

    if return_data
        return (data)
    else
        xyz2h5(data,xyz_h5,flight;
               tt_sort    = tt_sort,
               lines      = lines,
               lines_type = lines_type)
    end
end # function xyz2h5

"""
    xyz2h5(data::Array, xyz_h5::String, flight::Symbol;
           tt_sort::Bool      = true,
           lines::Vector      = [()],
           lines_type::Symbol = :exclude)
"""
function xyz2h5(data::Array, xyz_h5::String, flight::Symbol;
                tt_sort::Bool      = true,
                lines::Vector      = [()],
                lines_type::Symbol = :exclude)

    fields   = xyz_fields(flight) # vector of data field names
    Nf_chk   = length(fields)     # number of data fields (columns)
    ind_tt   = findfirst(fields .== :tt)   # time index (column)
    ind_line = findfirst(fields .== :line) # line index (column)

    # number of valid data rows & data fields
    (N,Nf) = size(data)

    @assert Nf == Nf_chk "xyz fields are of different dimensions, $N ≂̸ $Nf_chk"

    # check for duplicated data
    N_tt   = length(unique(data[:,ind_tt  ]))
    N_line = length(unique(data[:,ind_line]))
    N - N_tt > N_line && @info("xyz file may contain duplicated data")

    if isempty(lines[1])
        ind = trues(N)
    else

        # get ind for all lines
        ind = falses(N)
        for line in lines
            ind = ind .| get_ind(data[:,ind_tt],data[:,ind_line];
                                 lines=[line[1]],tt_lim=[line[2],line[3]])
        end

        # include or exclude lines
        if lines_type == :exclude
            ind .= .!ind
        elseif lines_type != :include
            error("$lines_type lines type not defined")
        end

    end

    # write N & dt data fields
    N  = sum(ind) # number of used data rows
    dt = N > 1 ? round(data[ind,ind_tt][2]-data[ind,ind_tt][1],digits=9) : 0.1 # measurement time step
    write_field(xyz_h5,:N ,N)
    write_field(xyz_h5,:dt,dt)

    ind_sort = tt_sort ? sortperm(data[ind,ind_tt]) : 1:N # sorting order

    # write other data fields
    for i = 1:Nf
        fields[i] != :ignore && write_field(xyz_h5,fields[i],
                                            data[ind,i][ind_sort,1])
    end

end # function xyz2h5

"""
    par(val::SubString{String})

Internal helper function to return `*` as `NaN`, otherwise parse as `Float64`.

**Arguments:**
- `val`: substring

**Returns:**
- `par`: parsed substring
"""
function par(val::SubString{String})
    val == "*" ? NaN : parse(Float64,val)
end # function par

"""
    remove_extension(data_file::String, extension::String)

Internal helper function to remove extension from path/name of data file. Checks
if `data_file` contains (case insensitive) `extension`, and if so, removes it.

**Arguments:**
- `data_file`: path/name of data file
- `extension`: extension to remove

**Returns:**
- `data_file`: path/name of data file, without `extension`
"""
function remove_extension(data_file::String, extension::String)
    f = data_file
    e = extension
    l = length(e)
    length(f) <= l && (return f)
    f = lowercase(f[end-l+1:end]) == lowercase(e) ? f[1:end-l] : f
end # function remove_extension

"""
    add_extension(data_file::String, extension::String)

Internal helper function to add extension to path/name of data file. Checks
if `data_file` contains (case insensitive) `extension`, and if not, adds it.

**Arguments:**
- `data_file`: path/name of data file
- `extension`: extension to add

**Returns:**
- `data_file`: path/name of data file, with `extension`
"""
function add_extension(data_file::String, extension::String)
    f = data_file
    e = extension
    l = length(e)
    length(f) <= l && (return f*e)
    f = lowercase(f[end-l+1:end]) == lowercase(e) ? f : f*e
end # function add_extension

"""
    delete_field(data_h5::String, field)

Delete a data field from an HDF5 file.

**Arguments:**
- `data_h5`: path/name of data HDF5 file (`.h5` extension optional)
- `field`:   data field in `data_h5` to delete

**Returns:**
- `nothing`: `field` is deleted in `data_h5`
"""
function delete_field(data_h5::String, field)
    data_h5 = add_extension(data_h5,".h5")
    field   = String(field)
    file    = h5open(data_h5,"r+") # read-write, preserve existing contents
    delete_object(file,field)
    close(file)
    return (nothing)
end # function delete_field

"""
    write_field(data_h5::String, field, data)

Write (add) a new data field and data in an HDF5 file.

**Arguments:**
- `data_h5`: path/name of data HDF5 file (`.h5` extension optional)
- `field`:   data field in `data_h5` to write
- `data`:    data to write

**Returns:**
- `nothing`: `field` with `data` is written in `data_h5`
"""
function write_field(data_h5::String, field, data)
    data_h5 = add_extension(data_h5,".h5")
    field   = String(field)
    h5open(data_h5,"cw") do file # read-write, create file if not existing, preserve existing contents
        write(file,field,data)
    end
    return (nothing)
end # function write_field

"""
    overwrite_field(data_h5::String, field, data)

Overwrite a data field and data in an HDF5 file.

**Arguments:**
- `data_h5`: path/name of data HDF5 file (`.h5` extension optional)
- `field`:   data field in `data_h5` to overwrite
- `data`:    data to write

**Returns:**
- `nothing`: `field` with `data` is written in `data_h5`
"""
function overwrite_field(data_h5::String, field, data)
    data_h5 = add_extension(data_h5,".h5")
    field   = String(field)
    delete_field(data_h5,field)
    write_field(data_h5,field,data)
    return (nothing)
end # function overwrite_field

"""
    read_field(data_h5::String, field)

Read data for a data field in an HDF5 file.

**Arguments:**
- `data_h5`: path/name of data HDF5 file (`.h5` extension optional)
- `field`:   data field in `data_h5` to read

**Returns:**
- `data`: data for `data field` in `data_h5`
"""
function read_field(data_h5::String, field)
    data_h5 = add_extension(data_h5,".h5")
    field   = String(field)
    h5open(data_h5,"r") do file # read-only
        read(file,field)
    end
end # function read_field

"""
    rename_field(data_h5::String, field_old, field_new)

Rename data field in an HDF5 file.

**Arguments:**
- `data_h5`:   path/name of data HDF5 file (`.h5` extension optional)
- `field_old`: old data field in `data_h5`
- `field_new`: new data field in `data_h5`

**Returns:**
- `nothing`: `field_old` is renamed `field_new` in `data_h5`
"""
function rename_field(data_h5::String, field_old, field_new)
    data_h5   = add_extension(data_h5,".h5")
    field_old = String(field_old)
    field_new = String(field_new)
    data      = read_field(data_h5,field_old)
    delete_field(data_h5,field_old)
    write_field(data_h5,field_new,data)
    return (nothing)
end # function rename_field

"""
    clear_fields(data_h5::String)

Clear all data fields and data in an HDF5 file.

**Arguments:**
- `data_h5`: path/name of data HDF5 file (`.h5` extension optional)

**Returns:**
- `nothing`: all data fields and data cleared in `data_h5`
"""
function clear_fields(data_h5::String)
    data_h5 = add_extension(data_h5,".h5")
    file    = h5open(data_h5,"cw") # read-write, create file if not existing, preserve existing contents
    close(file)
    file    = h5open(data_h5,"w") # read-write, destroy existing contents
    close(file)
    return (nothing)
end # function clear_fields

"""
    print_fields(s)

Print all data fields and types for a given struct.

**Arguments:**
- `s`: struct

**Returns:**
- `nothing`: all data fields and types in `s` are printed out
"""
function print_fields(s)
    for field in fieldnames(typeof(s))
        t = typeof(getfield(s,field))
        if parentmodule(t) == MagNav
            for f in fieldnames(t)
                println("$field.$f  ",typeof(getfield(getfield(s,field),f)))
            end
        else
            println("$field  ",t)
        end
    end
    return (nothing)
end # function print_fields

"""
    compare_fields(s1, s2; silent::Bool = false)

Compare data for each data field in 2 structs of the same type.

**Arguments:**
- `s1`:     struct 1
- `s2`:     struct 2
- `silent`: (optional) if true, no summary print out

**Returns:**
- `N_dif`: if `silent = false`, number of different fields
"""
function compare_fields(s1, s2; silent::Bool = false)
    t1 = typeof(s1)
    t2 = typeof(s2)
    @assert t1 == t2 "$t1 & $t2 types do no match"
    N_dif = 0;
    for field in fieldnames(t1)
        t  = typeof(getfield(s1,field))
        f1 = getfield(s1,field)
        f2 = getfield(s2,field)
        if parentmodule(t) == MagNav
            N_dif_add = compare_fields(f1,f2;silent=true)
            N_dif_add == 0 || println("($field is above)")
            N_dif += N_dif_add
        else
            if eltype(f1) <: Number
                if size(f1) != size(f2)
                    println("size of $field field is different")
                    N_dif += 1
                else
                    dif = sum(abs.(f1 - f2))
                    dif ≈ 0 || println("$field  ",dif)
                    dif ≈ 0 || (N_dif += 1)
                end
            elseif f1 isa Chain
                if length(f1) != length(f2)
                    println("size of $field field is different")
                    N_dif += 1
                else
                    for i in eachindex(f1)
                        for f in [:weight,:bias,:σ]
                            if getfield(f1[i],f) != getfield(f2[i],f)
                                println("layer $i $f field in $field is different")
                                N_dif += 1
                            end
                        end
                    end
                end
            else
                if f1 != f2
                    println("non-numeric $field field is different")
                    N_dif += 1
                end
            end
        end
    end

    if silent
        return (N_dif)
    else
        @info("number of different data fields: $N_dif")
    end
end # function compare_fields

"""
    field_check(s, t::Union{DataType,UnionAll})

Internal helper function to find data fields of a specified type in given struct.

**Arguments:**
- `s`: struct
- `t`: type

**Returns:**
- `fields`: data fields of type `t` in struct `s`
"""
function field_check(s, t::Union{DataType,UnionAll})
    fields = fieldnames(typeof(s))
    [fields[i] for i = findall([getfield(s,f) isa t for f in fields])]
end # function field_check

"""
    field_check(s, field::Symbol)

Internal helper function to check if a specified data field is in a given struct.

**Arguments:**
- `s`:     struct
- `field`: data field

**Returns:**
- `AssertionError` if `field` is not in struct `s`
"""
function field_check(s, field::Symbol)
    t = typeof(s)
    @assert field in fieldnames(t) "$field field not in $t type"
end # function field_check

"""
    field_check(s, field::Symbol, t::Union{DataType,UnionAll})

Internal helper function to check if a specified data field is in a given
struct and of a given type.

**Arguments:**
- `s`:     struct
- `field`: data field
- `t`:     type

**Returns:**
- `AssertionError` if `field` is not in struct `s` or not type `t`
"""
function field_check(s, field::Symbol, t::Union{DataType,UnionAll})
    field_check(s,field)
    @assert getfield(s,field) isa t "$field is not $t type"
end # function field_check

"""
    field_extrema(xyz::XYZ, field::Symbol, val)

Internal helper function to determine time extrema for specific value of data field.

**Arguments:**
- `xyz`:   `XYZ` flight data struct
- `field`: data field
- `val`:   specific value of `field`

**Returns:**
- `t_extrema`: time extrema for given field
"""
function field_extrema(xyz::XYZ, field::Symbol, val)
    if sum(getfield(xyz,field).==val) > 0
        extrema(xyz.traj.tt[getfield(xyz,field).==val])
    else
        error("$val not in $field")
    end
end # function field_extrema

"""
    xyz_fields(flight::Symbol)

Internal helper function to get field names for given SGL flight.
- Valid for SGL flights:
    - `:Flt1001`
    - `:Flt1002`
    - `:Flt1003`
    - `:Flt1004_1005`
    - `:Flt1004`
    - `:Flt1005`
    - `:Flt1006`
    - `:Flt1007`
    - `:Flt1008`
    - `:Flt1009`
    - `:Flt1001_160Hz`
    - `:Flt1002_160Hz`
    - `:Flt2001_2017`
    - `:Flt2001`
    - `:Flt2002`
    - `:Flt2004`
    - `:Flt2005`
    - `:Flt2006`
    - `:Flt2007`
    - `:Flt2008`
    - `:Flt2015`
    - `:Flt2016`
    - `:Flt2017`

**Arguments:**
- `flight`: flight name (e.g., `:Flt1001`)

**Returns:**
- `fields`: vector of data field names (Symbols)
"""
function xyz_fields(flight::Symbol)

    # get csv files containing fields from sgl_flight_data_fields artifact
    fields20  = sgl_fields(:fields_sgl_2020)
    fields21  = sgl_fields(:fields_sgl_2021)
    fields160 = sgl_fields(:fields_sgl_160)

    d = Dict{Symbol,Vector{Symbol}}()
    push!(d, :fields20  => Symbol.(vec(readdlm(fields20 ,','))))
    push!(d, :fields21  => Symbol.(vec(readdlm(fields21 ,','))))
    push!(d, :fields160 => Symbol.(vec(readdlm(fields160,','))))

    if flight in keys(d)

        return (d[flight])

    elseif flight in [:Flt1001,:Flt1002]

        # no mag_6_uc or flux_a for these flights
        exc = [:mag_6_uc,:flux_a_x,:flux_a_y,:flux_a_z,:flux_a_t]
        ind = .!(d[:fields20] .∈ (exc,))

        return (d[:fields20][ind])

    elseif flight in [:Flt1003,:Flt1004_1005,:Flt1004,:Flt1005,
                      :Flt1006,:Flt1007]

        # no mag_6_uc for these flights
        exc = [:mag_6_uc]
        ind = .!(d[:fields20] .∈ (exc,))

        return (d[:fields20][ind])

    elseif flight in [:Flt1008,:Flt1009]

        return (d[:fields20])

    elseif flight in [:Flt1001_160Hz,:Flt1002_160Hz]

        # no mag_6_uc or flux_a for these flights
        exc = [:mag_6_uc,:flux_a_x,:flux_a_y,:flux_a_z,:flux_a_t]
        ind = .!(d[:fields160] .∈ (exc,))

        return (d[:fields160][ind])

    elseif flight in [:Flt2001_2017,
                      :Flt2001,:Flt2002,:Flt2004,:Flt2005,:Flt2006,
                      :Flt2007,:Flt2008,:Flt2015,:Flt2016,:Flt2017]

        return (d[:fields21])

    else
        error("$flight flight not defined")
    end

end # function xyz_fields
