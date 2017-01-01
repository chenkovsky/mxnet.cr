@[Link("mxnet")]
lib LibMXNet
  alias MXUInt = UInt32
  alias MXFloat = Float32
  type NDArrayHandle = Void*
  type FunctionHandle = Void*
  type AtomicSymbolCreator = Void*
  type SymbolHandle = Void*
  type AtomicSymbolHandle = Void*
  type ExecutorHandle = Void*
  type DataIterCreator = Void*
  type DataIterHandle = Void*
  type KVStoreHandle = Void*
  type RecordIOHandle = Void*
  type RtcHandle = Void*
  type OptimizerCreator = Void*
  type OptimizerHandle = Void*
  type ExecutorMonitorCallback = UInt8*, NDArrayHandle, Void* -> Void
  type BucketKey = Void*

  struct NativeOpInfo
    forward : Int32, Float32**, Int32*, UInt16**, Int32*, Void* -> Void
    backward : Int32, Float32**, Int32*, UInt16**, Int32*, Void* -> Void
    infer_shape : Int32, Int32*, UInt16**, Void* -> Void
    list_outputs : UInt8***, Void* -> Void
    list_arguments : UInt8***, Void* -> Void
    p_forward : Void*
    p_backward : Void*
    p_infer_shape : Void*
    p_list_outputs : Void*
    p_list_arguments : Void*
  end

  struct NDArrayOpInfo
    forward : Int32, Void**, Int32*, Void* -> Bool
    backward : Int32, Void**, Int32*, Void* -> Bool
    infer_shape : Int32, Int32*, UInt16**, Void* -> Bool
    list_outputs : UInt8***, Void* -> Bool
    list_arguments : UInt8***, Void* -> Bool
    declare_backward_dependency : Int32*, Int32*, Int32*, Int32*, Int32**, Void* -> Bool
    p_forward : Void*
    p_backward : Void*
    p_infer_shape : Void*
    p_list_outputs : Void*
    p_list_arguments : Void*
    p_declare_backward_dependency : Void*
  end

  # return str message of the last error
  # all function in this file will return 0 when success
  # and -1 when an error occured,
  # MXGetLastError can be called to retrieve the error
  #
  # this function is threadsafe and can be called by different thread
  # return error info
  fun mx_get_last_error = MXGetLastError : UInt8*

  # -------------------------------------
  # Part 0: Global State setups
  # -------------------------------------

  # Seed the global random number generators in mxnet.
  # param seed the random number seed.
  # return 0 when success, -1 when failure happens.
  fun mx_random_seed = MXRandomSeed(seed : Int32) : Int32
  # Notify the engine about a shutdown,
  # This can help engine to print less messages into display.
  #
  # User do not have to call this function.
  # return 0 when success, -1 when failure happens.
  fun mx_notify_shutdown = MXNotifyShutdown : Int32

  # -------------------------------------
  # Part 1: NDArray creation and deletion
  # -------------------------------------
  # create a NDArray handle that is not initialized
  # can be used to pass in as mutate variables
  # to hold the result of NDArray
  # param out the returning handle
  # return 0 when success, -1 when failure happens
  fun mx_ndarray_create_none = MXNDArrayCreateNone(out_ : NDArrayHandle*) : Int32

  # create a NDArray with specified shape
  # param shape the pointer to the shape
  # param ndim the dimension of the shape
  # param dev_type device type, specify device we want to take
  # param dev_id the device id of the specific device
  # param delay_alloc whether to delay allocation until
  #       the narray is first mutated
  # param out the returning handle
  # return 0 when success, -1 when failure happens
  fun mx_ndarray_create = MXNDArrayCreate(shape : MXUInt*, ndim : MXUInt,
                                          dev_type : Int32, dev_id : Int32,
                                          delay_alloc : Int32, out_ : NDArrayHandle*) : Int32
  # create a NDArray with specified shape and data type
  # param shape the pointer to the shape
  # param ndim the dimension of the shape
  # param dev_type device type, specify device we want to take
  # param dev_id the device id of the specific device
  # param delay_alloc whether to delay allocation until
  #       the narray is first mutated
  # param dtype data type of created array
  # param out the returning handle
  # return 0 when success, -1 when failure happens
  fun mx_ndarray_create_ex = MXNDArrayCreateEx(shape : MXUInt*, ndim : MXUInt,
                                               dev_type : Int32, dev_id : Int32,
                                               delay_alloc : Int32, dtype : Int32,
                                               out_ : NDArrayHandle*) : Int32
  # create a NDArray handle that is loaded from raw bytes.
  # param buf the head of the raw bytes
  # param size size of the raw bytes
  # param out the returning handle
  # return 0 when success, -1 when failure happens
  fun mx_ndarray_load_from_raw_bytes = MXNDArrayLoadFromRawBytes(buf : Void*, size : LibC::SizeT, out_ : NDArrayHandle*) : Int32
  # save the NDArray into raw bytes.
  # param handle the NDArray handle
  # param out_size size of the raw bytes
  # param out_buf the head of returning memory bytes.
  # return 0 when success, -1 when failure happens
  fun mx_ndarray_save_raw_bytes = MXNDArraySaveRawBytes(handle : NDArrayHandle, out_size : LibC::SizeT*, out_buf : UInt8**) : Int32

  # Save list of narray into the file.
  # param fname name of the file.
  # param num_args number of arguments to save.
  # param args the array of NDArrayHandles to be saved.
  # param keys the name of the NDArray, optional, can be NULL
  # return 0 when success, -1 when failure happens
  fun mx_ndarray_save = MXNDArraySave(fname : UInt8*, num_args : MXUInt, args : NDArrayHandle*, keys : UInt8**) : Int32
  # Load list of narray from the file.
  # param fname name of the file.
  # param out_size number of narray loaded.
  # param out_arr head of the returning narray handles.
  # param out_name_size size of output name arrray.
  # param out_names the names of returning NDArrays, can be NULL
  # return 0 when success, -1 when failure happens
  fun mx_ndarray_load = MXNDArrayLoad(fname : UInt8*, out_size : MXUInt*, out_arr : NDArrayHandle**, out_name_size : MXUInt*, out_names : UInt8***) : Int32
  # Perform a synchronize copy from a continugous CPU memory region.
  #
  # This function will call WaitToWrite before the copy is performed.
  # This is useful to copy data from existing memory region that are
  # not wrapped by NDArray(thus dependency not being tracked).
  #
  # param handle the NDArray handle
  # param data the data source to copy from.
  # param size the memory size we want to copy from.
  fun mx_ndarray_sync_copy_from_cpu = MXNDArraySyncCopyFromCPU(handle : NDArrayHandle, data : Void*, size : LibC::SizeT) : Int32
  #  Perform a synchronize copyto a continugous CPU memory region.
  #
  # This function will call WaitToRead before the copy is performed.
  # This is useful to copy data from existing memory region that are
  # not wrapped by NDArray(thus dependency not being tracked).
  #
  # param handle the NDArray handle
  # param data the data source to copy into.
  # param size the memory size we want to copy into.
  fun mx_ndarray_sync_copy_to_cpu = MXNDArraySyncCopyToCPU(handle : NDArrayHandle, data : Void*, size : LibC::SizeT) : Int32

  # Wait until all the pending writes with respect NDArray are finished.
  # Always call this before read data out synchronizely.
  # param handle the NDArray handle
  # return 0 when success, -1 when failure happens
  fun mx_ndarray_wait_to_read = MXNDArrayWaitToRead(handle : NDArrayHandle) : Int32
  # Wait until all the pending read/write with respect NDArray are finished.
  # Always call this before write data into NDArray synchronizely.
  # param handle the NDArray handle
  # return 0 when success, -1 when failure happens
  fun mx_ndarray_wait_to_write = MXNDArrayWaitToWrite(handle : NDArrayHandle) : Int32
  # free the narray handle
  # param handle the handle to be freed
  # return 0 when success, -1 when failure happens
  fun mx_ndarray_free = MXNDArrayFree(handle : NDArrayHandle) : Int32
  # Slice the NDArray along axis 0.
  # param handle the handle to the narraya
  # param slice_begin The beginning index of slice
  # param slice_end The ending index of slice
  # param out The NDArrayHandle of sliced NDArray
  # return 0 when success, -1 when failure happens
  fun mx_ndarray_slice = MXNDArraySlice(handle : NDArrayHandle, slice_begin : MXUInt,
                                        slice_end : MXUInt, out_ : NDArrayHandle*) : Int32
  # Reshape the NDArray.
  # param handle the handle to the narray
  # param ndim number of dimensions of new shape
  # param dims new shape
  # param out the NDArrayHandle of reshaped NDArray
  # return 0 when success, -1 when failure happens
  fun mx_ndarray_reshape = MXNDArrayReshape(handle : NDArrayHandle, ndim : Int32,
                                            dims : Int32*, out_ : NDArrayHandle*) : Int32
  # get the shape of the array
  # param handle the handle to the narray
  # param out_dim the output dimension
  # param out_pdata pointer holder to get data pointer of the shape
  # return 0 when success, -1 when failure happens
  fun mx_ndarray_get_shape = MXNDArrayGetShape(handle : NDArrayHandle, out_dim : MXUInt*, out_pdata : MXUInt**) : Int32
  # get the content of the data in NDArray
  # param handle the handle to the narray
  # param out_pdata pointer holder to get pointer of data
  # return 0 when success, -1 when failure happens
  fun mx_ndarray_get_data = MXNDArrayGetData(handle : NDArrayHandle, out_pdata : MXFloat**) : Int32
  # get the type of the data in NDArray
  # param handle the handle to the narray
  # param out_dtype pointer holder to get type of data
  # return 0 when success, -1 when failure happens
  fun mx_ndarray_get_dtype = MXNDArrayGetDType(handle : NDArrayHandle, out_dtype : Int32*) : Int32
  # get the context of the NDArray
  # param handle the handle to the narray
  # param out_dev_type the output device type
  # param out_dev_id the output device id
  # return 0 when success, -1 when failure happens
  fun mx_ndarray_get_context = MXNDArrayGetContext(handle : NDArrayHandle, out_dev_type : Int32*, out_dev_id : Int32*) : Int32

  # --------------------------------
  # Part 2: functions on NDArray
  # --------------------------------
  # list all the available functions handles
  # most user can use it to list all the needed functions
  # param out_size the size of returned array
  # param out_array the output function array
  # return 0 when success, -1 when failure happens
  fun mx_list_functions = MXListFunctions(out_size : MXUInt*, out_array : FunctionHandle**) : Int32

  # get the function handle by name
  # param name the name of the function
  # param out the corresponding function handle
  # return 0 when success, -1 when failure happens
  fun mx_get_function = MXGetFunction(name : UInt8*, out_ : FunctionHandle*) : Int32

  # Get the information of the function handle.
  # param fun The function handle.
  # param name The returned name of the function.
  # param description The returned description of the function.
  # param num_args Number of arguments.
  # param arg_names Name of the arguments.
  # param arg_type_infos Type informations about the arguments.
  # param arg_descriptions Description information about the arguments.
  # param return_type Return type of the function.
  # return 0 when success, -1 when failure happens
  fun mx_func_get_info = MXFuncGetInfo(fun_ : FunctionHandle, name : UInt8**,
                                       description : UInt8**, num_args : MXUInt*,
                                       arg_names : UInt8***, arg_type_infos : UInt8***,
                                       arg_descriptions : UInt8***, return_type : UInt8**) : Int32
  # get the argument requirements of the function
  # param fun input function handle
  # param num_use_vars how many NDArrays to be passed in as used_vars
  # param num_scalars scalar variable is needed
  # param num_mutate_vars how many NDArrays to be passed in as mutate_vars
  # param type_mask the type mask of this function
  # return 0 when success, -1 when failure happens
  # sa MXFuncInvoke
  fun mx_func_describe = MXFuncDescribe(fun_ : FunctionHandle, num_use_vars : MXUInt*,
                                        num_scalars : MXUInt*, num_mutate_vars : MXUInt*,
                                        type_mask : Int32*) : Int32
  # invoke a function, the array size of passed in arguments
  # must match the values in the
  # param fun the function
  # param use_vars the normal arguments passed to function
  # param scalar_args the scalar qarguments
  # param mutate_vars the mutate arguments
  # return 0 when success, -1 when failure happens
  # sa MXFuncDescribeArgs
  fun mx_func_invoke = MXFuncInvoke(fun_ : FunctionHandle, use_vars : NDArrayHandle*,
                                    scalar_args : MXFloat*, mutate_vars : NDArrayHandle*) : Int32
  # invoke a function, the array size of passed in arguments
  # must match the values in the
  # param fun the function
  # param use_vars the normal arguments passed to function
  # param scalar_args the scalar qarguments
  # param mutate_vars the mutate arguments
  # param num_params number of keyword parameters
  # param param_keys keys for keyword parameters
  # param param_vals values for keyword parameters
  # return 0 when success, -1 when failure happens
  # sa MXFuncDescribeArgs
  fun mx_func_invoke_ex = MXFuncInvokeEx(fun_ : FunctionHandle, use_vars : NDArrayHandle*,
                                         scalar_args : MXFloat*, mutate_vars : NDArrayHandle*,
                                         num_params : Int32, param_keys : UInt8**,
                                         param_vals : UInt8**) : Int32

  # --------------------------------------------
  # Part 3: symbolic configuration generation
  # --------------------------------------------
  # list all the available AtomicSymbolEntry
  # param out_size the size of returned array
  # param out_array the output AtomicSymbolCreator array
  # return 0 when success, -1 when failure happens
  fun mx_symbol_list_atomic_symbol_creators = MXSymbolListAtomicSymbolCreators(out_size : MXUInt*, out_array : AtomicSymbolCreator**) : Int32

  # Get the detailed information about atomic symbol.
  # param creator the AtomicSymbolCreator.
  # param name The returned name of the creator.
  # param description The returned description of the symbol.
  # param num_args Number of arguments.
  # param arg_names Name of the arguments.
  # param arg_type_infos Type informations about the arguments.
  # param arg_descriptions Description information about the arguments.
  # param key_var_num_args The keyword argument for specifying variable number of arguments.
  #           When this parameter has non-zero length, the function allows variable number
  #           of positional arguments, and will need the caller to pass it in in
  #           MXSymbolCreateAtomicSymbol,
  #           With key = key_var_num_args, and value = number of positional arguments.
  # param return_type Return type of the function, can be Symbol or Symbol[]
  # return 0 when success, -1 when failure happens
  fun mx_symbol_get_atomic_symbol_info = MXSymbolGetAtomicSymbolInfo(creator : AtomicSymbolCreator, name : UInt8**,
                                                                     description : UInt8**, num_args : MXUInt*,
                                                                     arg_names : UInt8***, arg_type_infos : UInt8***,
                                                                     arg_descriptions : UInt8***, key_var_num_args : UInt8**,
                                                                     return_type : UInt8**) : Int32

  # Create an AtomicSymbol.
  # param creator the AtomicSymbolCreator
  # param num_param the number of parameters
  # param keys the keys to the params
  # param vals the vals of the params
  # param out pointer to the created symbol handle
  # return 0 when success, -1 when failure happens
  fun mx_symbol_create_atomic_symbol = MXSymbolCreateAtomicSymbol(creator : AtomicSymbolCreator, num_param : MXUInt, keys : UInt8**, vals : UInt8**, out_ : SymbolHandle*) : Int32

  # Create a Variable Symbol.
  # param name name of the variable
  # param out pointer to the created symbol handle
  # return 0 when success, -1 when failure happens
  fun mx_symbol_create_variable = MXSymbolCreateVariable(name : UInt8*, out_ : SymbolHandle*) : Int32
  # Create a Symbol by grouping list of symbols together
  # param num_symbols number of symbols to be grouped
  # param symbols array of symbol handles
  # param out pointer to the created symbol handle
  # return 0 when success, -1 when failure happens
  fun mx_symbol_create_group = MXSymbolCreateGroup(num_symbols : MXUInt, symbols : SymbolHandle*, out_ : SymbolHandle*) : Int32
  # Load a symbol from a json file.
  # param fname the file name.
  # param out the output symbol.
  # return 0 when success, -1 when failure happens
  fun mx_symbol_create_from_file = MXSymbolCreateFromFile(fname : UInt8*, out_ : SymbolHandle*) : Int32
  # Load a symbol from a json string.
  # param json the json string.
  # param out the output symbol.
  # return 0 when success, -1 when failure happens
  fun mx_symbol_create_from_json = MXSymbolCreateFromJSON(json : UInt8*, out_ : SymbolHandle*) : Int32
  # Save a symbol into a json file.
  # param symbol the input symbol.
  # param fname the file name.
  # return 0 when success, -1 when failure happens
  fun mx_symbol_save_to_file = MXSymbolSaveToFile(symbol : SymbolHandle, fname : UInt8*) : Int32

  # Save a symbol into a json string
  # param symbol the input symbol.
  # param out_json output json string.
  # return 0 when success, -1 when failure happens
  fun mx_symbol_save_to_json = MXSymbolSaveToJSON(symbol : SymbolHandle, out_json : UInt8**) : Int32
  # Free the symbol handle.
  # param symbol the symbol
  # return 0 when success, -1 when failure happens
  fun mx_symbol_free = MXSymbolFree(symbol : SymbolHandle) : Int32

  # Copy the symbol to another handle
  # param symbol the source symbol
  # param out used to hold the result of copy
  # return 0 when success, -1 when failure happens
  fun mx_symbol_copy = MXSymbolCopy(symbol : SymbolHandle, out_ : SymbolHandle*) : Int32
  # brief Print the content of symbol, used for debug.
  # param symbol the symbol
  # param out_str pointer to hold the output string of the printing.
  # return 0 when success, -1 when failure happens
  fun mx_symbol_print = MXSymbolPrint(symbol : SymbolHandle, out_str : UInt8**) : Int32
  # Get string name from symbol
  # param symbol the source symbol
  # param out The result name.
  # param success Whether the result is contained in out.
  # return 0 when success, -1 when failure happens
  fun mx_symbol_get_name = MXSymbolGetName(symbol : SymbolHandle, out_ : UInt8**, success : Int32*) : Int32
  # Get string attribute from symbol
  # param symbol the source symbol
  # param key The key of the symbol.
  # param out The result attribute, can be NULL if the attribute do not exist.
  # param success Whether the result is contained in out.
  # return 0 when success, -1 when failure happens
  fun mx_symbol_get_attr = MXSymbolGetAttr(symbol : SymbolHandle, key : UInt8*, out_ : UInt8**, success : Int32*) : Int32
  #  Set string attribute from symbol.
  # NOTE: Setting attribute to a symbol can affect the semantics(mutable/immutable) of symbolic graph.
  #
  # Safe recommendaton: use  immutable graph
  # - Only allow set attributes during creation of new symbol as optional parameter
  #
  # Mutable graph (be careful about the semantics):
  # - Allow set attr at any point.
  # - Mutating an attribute of some common node of two graphs can cause confusion from user.
  #
  # param symbol the source symbol
  # param key The key of the symbol.
  # param value The value to be saved.
  # return 0 when success, -1 when failure happens
  fun mx_symbol_set_attr = MXSymbolSetAttr(symbol : SymbolHandle, key : UInt8*, value : UInt8*) : Int32
  # Get all attributes from symbol
  # param symbol the source symbol
  # param out_size The number of output attributes
  # param out 2*out_size strings representing key value pairs.
  # return 0 when success, -1 when failure happens
  fun mx_symbol_list_attr = MXSymbolListAttr(symbol : SymbolHandle, out_size : MXUInt*, out_ : UInt8***) : Int32
  fun mx_symbol_list_attr_shallow = MXSymbolListAttrShallow(symbol : SymbolHandle, out_size : MXUInt*, out_ : UInt8***) : Int32
  # List arguments in the symbol.
  # param symbol the symbol
  # param out_size output size
  # param out_str_array pointer to hold the output string array
  # return 0 when success, -1 when failure happens
  fun mx_symbol_list_arguments = MXSymbolListArguments(symbol : SymbolHandle, out_size : MXUInt*, out_str_array : UInt8***) : Int32
  # List returns in the symbol.
  # param symbol the symbol
  # param out_size output size
  # param out_str_array pointer to hold the output string array
  # return 0 when success, -1 when failure happens
  fun mx_symbol_list_outputs = MXSymbolListOutputs(symbol : SymbolHandle, out_size : MXUInt*, out_str_array : UInt8***) : Int32
  # Get a symbol that contains all the internals.
  # param symbol The symbol
  # param out The output symbol whose outputs are all the internals.
  # return 0 when success, -1 when failure happens
  fun mx_symbol_get_internals = MXSymbolGetInternals(symbol : SymbolHandle, out_ : SymbolHandle*) : Int32
  # Get index-th outputs of the symbol.
  # param symbol The symbol
  # param index the Index of the output.
  # param out The output symbol whose outputs are the index-th symbol.
  # return 0 when success, -1 when failure happens
  fun mx_symbol_get_output = MXSymbolGetOutput(symbol : SymbolHandle, index : MXUInt, out_ : SymbolHandle*) : Int32
  # List auxiliary states in the symbol.
  # param symbol the symbol
  # param out_size output size
  # param out_str_array pointer to hold the output string array
  # return 0 when success, -1 when failure happens
  fun mx_symbol_list_auxiliary_states = MXSymbolListAuxiliaryStates(symbol : SymbolHandle, out_size : MXUInt*, out_str_array : UInt8***) : Int32
  # Compose the symbol on other symbols.
  #
  # This function will change the sym hanlde.
  # To achieve function apply behavior, copy the symbol first
  # before apply.
  #
  # param sym the symbol to apply
  # param name the name of symbol
  # param num_args number of arguments
  # param keys the key of keyword args (optional)
  # param args arguments to sym
  # return 0 when success, -1 when failure happens
  fun mx_symbol_compose = MXSymbolCompose(sym : SymbolHandle, name : UInt8*, num_args : MXUInt, keys : UInt8**, args : SymbolHandle*) : Int32
  # Get the gradient graph of the symbol
  #
  # param sym the symbol to get gradient
  # param num_wrt number of arguments to get gradient
  # param wrt the name of the arguments to get gradient
  # param out the returned symbol that has gradient
  # return 0 when success, -1 when failure happens
  fun mx_symbol_grad = MXSymbolGrad(sym : SymbolHandle, num_wrt : MXUInt, wrt : UInt8**, out_ : SymbolHandle*) : Int32

  # infer shape of unknown input shapes given the known one.
  # The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data
  # The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.
  #
  # param sym symbol handle
  # param num_args numbe of input arguments.
  # param keys the key of keyword args (optional)
  # param arg_ind_ptr the head pointer of the rows in CSR
  # param arg_shape_data the content of the CSR
  # param in_shape_size sizeof the returning array of in_shapes
  # param in_shape_ndim returning array of shape dimensions of eachs input shape.
  # param in_shape_data returning array of pointers to head of the input shape.
  # param out_shape_size sizeof the returning array of out_shapes
  # param out_shape_ndim returning array of shape dimensions of eachs input shape.
  # param out_shape_data returning array of pointers to head of the input shape.
  # param aux_shape_size sizeof the returning array of aux_shapes
  # param aux_shape_ndim returning array of shape dimensions of eachs auxiliary shape.
  # param aux_shape_data returning array of pointers to head of the auxiliary shape.
  # param complete whether infer shape completes or more information is needed.
  # return 0 when success, -1 when failure happens
  fun mx_symbol_infer_shape = MXSymbolInferShape(sym : SymbolHandle, num_args : MXUInt, keys : UInt8**,
                                                 arg_ind_ptr : MXUInt*, arg_shape_data : MXUInt*,
                                                 in_shape_size : MXUInt*, in_shape_ndim : MXUInt**,
                                                 in_shape_data : MXUInt***, out_shape_size : MXUInt*,
                                                 out_shape_ndim : MXUInt**, out_shape_data : MXUInt***,
                                                 aux_shape_size : MXUInt*, aux_shape_ndim : MXUInt**,
                                                 aux_shape_data : MXUInt***, complete : Int32*) : Int32

  # partially infer shape of unknown input shapes given the known one.
  #  *
  #  Return partially inferred results if not all shapes could be inferred.
  #  The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data
  #  The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.
  #  *
  # param sym symbol handle
  # param num_args numbe of input arguments.
  # param keys the key of keyword args (optional)
  # param arg_ind_ptr the head pointer of the rows in CSR
  # param arg_shape_data the content of the CSR
  # param in_shape_size sizeof the returning array of in_shapes
  # param in_shape_ndim returning array of shape dimensions of eachs input shape.
  # param in_shape_data returning array of pointers to head of the input shape.
  # param out_shape_size sizeof the returning array of out_shapes
  # param out_shape_ndim returning array of shape dimensions of eachs input shape.
  # param out_shape_data returning array of pointers to head of the input shape.
  # param aux_shape_size sizeof the returning array of aux_shapes
  # param aux_shape_ndim returning array of shape dimensions of eachs auxiliary shape.
  # param aux_shape_data returning array of pointers to head of the auxiliary shape.
  # param complete whether infer shape completes or more information is needed.
  # return 0 when success, -1 when failure happens
  fun mx_symbol_infer_shape_partial = MXSymbolInferShapePartial(sym : SymbolHandle, num_args : MXUInt,
                                                                keys : UInt8**, arg_ind_ptr : MXUInt*,
                                                                arg_shape_data : MXUInt*,
                                                                in_shape_size : MXUInt*,
                                                                in_shape_ndim : MXUInt**, in_shape_data : MXUInt***,
                                                                out_shape_size : MXUInt*,
                                                                out_shape_ndim : MXUInt**, out_shape_data : MXUInt***,
                                                                aux_shape_size : MXUInt*,
                                                                aux_shape_ndim : MXUInt**, aux_shape_data : MXUInt***,
                                                                complete : Int32*) : Int32
  # infer type of unknown input types given the known one.
  # The types are packed into a CSR matrix represented by arg_ind_ptr and arg_type_data
  # The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.
  # *
  # param sym symbol handle
  # param num_args numbe of input arguments.
  # param keys the key of keyword args (optional)
  # param arg_type_data the content of the CSR
  # param in_type_size sizeof the returning array of in_types
  # param in_type_data returning array of pointers to head of the input type.
  # param out_type_size sizeof the returning array of out_types
  # param out_type_data returning array of pointers to head of the input type.
  # param aux_type_size sizeof the returning array of aux_types
  # param aux_type_data returning array of pointers to head of the auxiliary type.
  # param complete whether infer type completes or more information is needed.
  # return 0 when success, -1 when failure happens
  fun mx_symbol_infer_type = MXSymbolInferType(sym : SymbolHandle, num_args : MXUInt,
                                               keys : UInt8**, arg_type_data : Int32*,
                                               in_type_size : MXUInt*, in_type_data : Int32**,
                                               out_type_size : MXUInt*, out_type_data : Int32**,
                                               aux_type_size : MXUInt*, aux_type_data : Int32**,
                                               complete : Int32*) : Int32
  # --------------------------------------------
  # Part 4: Executor interface
  # --------------------------------------------
  # Delete the executor
  # param handle the executor.
  # return 0 when success, -1 when failure happens
  fun mx_executor_free = MXExecutorFree(handle : ExecutorHandle) : Int32
  # Print the content of execution plan, used for debug.
  # param handle the executor.
  # param out_str pointer to hold the output string of the printing.
  # return 0 when success, -1 when failure happens
  fun mx_executor_print = MXExecutorPrint(handle : ExecutorHandle, out_str : UInt8**) : Int32
  # Executor forward method
  #
  # param handle executor handle
  # param is_train bool value to indicate whether the forward pass is for evaluation
  # return 0 when success, -1 when failure happens
  fun mx_executor_forward = MXExecutorForward(handle : ExecutorHandle, is_train : Int32) : Int32
  # Excecutor run backward
  #
  # param handle execute handle
  # param len lenth
  # param head_grads NDArray handle for heads' gradient
  #
  # return 0 when success, -1 when failure happens
  fun mx_executor_backward = MXExecutorBackward(handle : ExecutorHandle, len : MXUInt, head_grads : NDArrayHandle*) : Int32

  # Get executor's head NDArray
  #
  # param handle executor handle
  # param out_size output narray vector size
  # param out out put narray handles
  # return 0 when success, -1 when failure happens
  fun mx_executor_outputs = MXExecutorOutputs(handle : ExecutorHandle, out_size : MXUInt*, out_ : NDArrayHandle**) : Int32

  # Generate Executor from symbol
  #
  # param symbol_handle symbol handle
  # param dev_type device type
  # param dev_id device id
  # param len length
  # param in_args in args array
  # param arg_grad_store arg grads handle array
  # param grad_req_type grad req array
  # param aux_states_len length of auxiliary states
  # param aux_states auxiliary states array
  # param out output executor handle
  # return 0 when success, -1 when failure happens
  fun mx_executor_bind = MXExecutorBind(symbol_handle : SymbolHandle, dev_type : Int32,
                                        dev_id : Int32, len : MXUInt,
                                        in_args : NDArrayHandle*, arg_grad_store : NDArrayHandle*,
                                        grad_req_type : MXUInt*, aux_states_len : MXUInt,
                                        aux_states : NDArrayHandle*, out_ : ExecutorHandle*) : Int32
  # Generate Executor from symbol,
  # This is advanced function, allow specify group2ctx map.
  # The user can annotate "ctx_group" attribute to name each group.
  #
  # param symbol_handle symbol handle
  # param dev_type device type of default context
  # param dev_id device id of default context
  # param num_map_keys size of group2ctx map
  # param map_keys keys of group2ctx map
  # param map_dev_types device type of group2ctx map
  # param map_dev_ids device id of group2ctx map
  # param len length
  # param in_args in args array
  # param arg_grad_store arg grads handle array
  # param grad_req_type grad req array
  # param aux_states_len length of auxiliary states
  # param aux_states auxiliary states array
  # param out output executor handle
  # return 0 when success, -1 when failure happens
  fun mx_executor_bindx = MXExecutorBindX(symbol_handle : SymbolHandle, dev_type : Int32,
                                          dev_id : Int32, num_map_keys : MXUInt,
                                          map_keys : UInt8**, map_dev_types : Int32*,
                                          map_dev_ids : Int32*, len : MXUInt,
                                          in_args : NDArrayHandle*, arg_grad_store : NDArrayHandle*,
                                          grad_req_type : MXUInt, aux_states_len : MXUInt,
                                          aux_states : NDArrayHandle*, out_ : ExecutorHandle*) : Int32
  # Generate Executor from symbol,
  # This is advanced function, allow specify group2ctx map.
  # The user can annotate "ctx_group" attribute to name each group.
  #
  # param symbol_handle symbol handle
  # param dev_type device type of default context
  # param dev_id device id of default context
  # param num_map_keys size of group2ctx map
  # param map_keys keys of group2ctx map
  # param map_dev_types device type of group2ctx map
  # param map_dev_ids device id of group2ctx map
  # param len length
  # param in_args in args array
  # param arg_grad_store arg grads handle array
  # param grad_req_type grad req array
  # param aux_states_len length of auxiliary states
  # param aux_states auxiliary states array
  # param shared_exec input executor handle for memory sharing
  # param out output executor handle
  # return 0 when success, -1 when failure happens
  fun mx_executor_bind_ex = MXExecutorBindEX(symbol_handle : SymbolHandle, dev_type : Int32,
                                             dev_id : Int32, num_map_keys : MXUInt,
                                             map_keys : UInt8**, map_dev_types : Int32*,
                                             map_dev_ids : Int32*, len : MXUInt,
                                             in_args : NDArrayHandle*, arg_grad_store : NDArrayHandle*,
                                             grad_req_type : MXUInt*, aux_states_len : MXUInt,
                                             aux_states : NDArrayHandle*, shared_exec : Void*,
                                             out_ : ExecutorHandle*) : Int32

  # set a call back to notify the completion of operation
  fun mx_executor_set_monitor_callback = MXExecutorSetMonitorCallback(handle : ExecutorHandle,
                                                                      callback : ExecutorMonitorCallback,
                                                                      callback_handle : Void*) : Int32

  # --------------------------------------------
  # Part 5: IO Interface
  # --------------------------------------------
  # List all the available iterator entries
  # param out_size the size of returned iterators
  # param out_array the output iteratos entries
  # return 0 when success, -1 when failure happens
  fun mx_list_data_iters = MXListDataIters(out_size : MXUInt*, out_array : DataIterCreator**) : Int32

  # Init an iterator, init with parameters
  # the array size of passed in arguments
  # param handle of the iterator creator
  # param num_param number of parameter
  # param keys parameter keys
  # param vals parameter values
  # param out resulting iterator
  # return 0 when success, -1 when failure happens
  fun mx_data_iter_create_iter = MXDataIterCreateIter(handle : DataIterCreator,
                                                      num_param : MXUInt,
                                                      keys : UInt8**,
                                                      vals : UInt8**,
                                                      out_ : DataIterHandle*) : Int32
  # Get the detailed information about data iterator.
  # param creator the DataIterCreator.
  # param name The returned name of the creator.
  # param description The returned description of the symbol.
  # param num_args Number of arguments.
  # param arg_names Name of the arguments.
  # param arg_type_infos Type informations about the arguments.
  # param arg_descriptions Description information about the arguments.
  # return 0 when success, -1 when failure happens
  fun mx_data_iter_get_iter_info = MXDataIterGetIterInfo(creator : DataIterCreator,
                                                         name : UInt8**,
                                                         description : UInt8**,
                                                         num_args : MXUInt*,
                                                         arg_names : UInt8***,
                                                         arg_type_infos : UInt8***,
                                                         arg_descriptions : UInt8***) : Int32
  # Free the handle to the IO module
  # param handle the handle pointer to the data iterator
  # return 0 when success, -1 when failure happens
  fun mx_data_iter_free = MXDataIterFree(handle : DataIterHandle) : Int32

  # Move iterator to next position
  # param handle the handle to iterator
  # param out return value of next
  # return 0 when success, -1 when failure happens
  fun mx_data_iter_next = MXDataIterNext(handle : DataIterHandle, out_ : Int32*) : Int32
  # Call iterator.Reset
  # param handle the handle to iterator
  # return 0 when success, -1 when failure happens
  fun mx_data_iter_before_first = MXDataIterBeforeFirst(handle : DataIterHandle) : Int32
  # Get the handle to the NDArray of underlying data
  # param handle the handle pointer to the data iterator
  # param out handle to underlying data NDArray
  # return 0 when success, -1 when failure happens
  fun mx_data_iter_get_data = MXDataIterGetData(handle : DataIterHandle, out_ : NDArrayHandle*) : Int32
  # Get the image index by array.
  # param handle the handle pointer to the data iterator
  # param out_index output index of the array.
  # param out_size output size of the array.
  # return 0 when success, -1 when failure happens
  fun mx_data_iter_get_index = MXDataIterGetIndex(handle : DataIterHandle, out_index : UInt64**, out_size : UInt64*) : Int32
  # Get the padding number in current data batch
  # param handle the handle pointer to the data iterator
  # param pad pad number ptr
  # return 0 when success, -1 when failure happens
  fun mx_data_iter_get_pad_num = MXDataIterGetPadNum(handle : DataIterHandle, pad : Int32*) : Int32
  # Get the handle to the NDArray of underlying label
  # param handle the handle pointer to the data iterator
  # param out the handle to underlying label NDArray
  # return 0 when success, -1 when failure happens
  fun mx_data_iter_get_label = MXDataIterGetLabel(handle : DataIterHandle, out_ : NDArrayHandle*) : Int32

  # --------------------------------------------
  # Part 6: basic KVStore interface
  # --------------------------------------------
  # Initialized ps-lite environment variables
  # param num_vars number of variables to initialize
  # param keys environment keys
  # param vals environment values
  fun mx_init_ps_env = MXInitPSEnv(num_vars : MXUInt, keys : UInt8**, vals : UInt8**) : Int32
  # Create a kvstore
  # param type the type of KVStore
  # param out The output type of KVStore
  # return 0 when success, -1 when failure happens
  fun mx_kv_store_create = MXKVStoreCreate(type_ : UInt8*, out_ : KVStoreHandle*) : Int32
  # Delete a KVStore handle.
  # param handle handle to the kvstore
  # return 0 when success, -1 when failure happens
  fun mx_kv_store_free = MXKVStoreFree(handle : KVStoreHandle) : Int32
  # Init a list of (key,value) pairs in kvstore
  # param handle handle to the kvstore
  # param num the number of key-value pairs
  # param keys the list of keys
  # param vals the list of values
  # return 0 when success, -1 when failure happens
  fun mx_kv_store_init = MXKVStoreInit(handle : KVStoreHandle, num : MXUInt, keys : Int32*, vals : NDArrayHandle*) : Int32

  # Push a list of (key,value) pairs to kvstore
  # param handle handle to the kvstore
  # param num the number of key-value pairs
  # param keys the list of keys
  # param vals the list of values
  # param priority the priority of the action
  # return 0 when success, -1 when failure happens
  fun mx_kv_store_push = MXKVStorePush(handle : KVStoreHandle, num : MXUInt, keys : Int32*, vals : NDArrayHandle*, priority : Int32) : Int32
  # pull a list of (key, value) pairs from the kvstore
  # param handle handle to the kvstore
  # param num the number of key-value pairs
  # param keys the list of keys
  # param vals the list of values
  # param priority the priority of the action
  # return 0 when success, -1 when failure happens
  fun mx_kv_store_pull = MXKVStorePull(handle : KVStoreHandle, num : MXUInt, keys : Int32*, vals : NDArrayHandle*, priority : Int32) : Int32
  # user-defined updater for the kvstore
  # It's this updater's responsibility to delete \a recv and \a local
  # param the key
  # param recv the pushed value on this key
  # param local the value stored on local on this key
  # param handle The additional handle to the updater
  type MXKVStoreUpdater = Int32, NDArrayHandle, NDArrayHandle, Void* -> Void
  # register an push updater
  # param handle handle to the KVStore
  # param updater udpater function
  # param updater_handle The additional handle used to invoke the updater
  # return 0 when success, -1 when failure happens
  fun mx_kv_store_set_updater = MXKVStoreSetUpdater(handle : KVStoreHandle, updater : MXKVStoreUpdater, updater_handle : Void*) : Int32

  # get the type of the kvstore
  # param handle handle to the KVStore
  # param type a string type
  # return 0 when success, -1 when failure happens
  fun mx_kv_store_get_type = MXKVStoreGetType(handle : KVStoreHandle, type_ : UInt8**) : Int32
  # --------------------------------------------
  # Part 6: advanced KVStore for multi-machines
  # --------------------------------------------
  # return The rank of this node in its group, which is in [0, GroupSize).
  #
  # param handle handle to the KVStore
  # param ret the node rank
  # return 0 when success, -1 when failure happens
  fun mx_kv_store_get_rank = MXKVStoreGetRank(handle : KVStoreHandle, ret : Int32*) : Int32
  # return The number of nodes in this group, which is
  # - number of workers if if `IsWorkerNode() == true`,
  # - number of servers if if `IsServerNode() == true`,
  # - 1 if `IsSchedulerNode() == true`,
  # param handle handle to the KVStore
  # param ret the group size
  # return 0 when success, -1 when failure happens
  fun mx_kv_store_get_group_size = MXKVStoreGetGroupSize(handle : KVStoreHandle, ret : Int32*) : Int32

  # return whether or not this process is a worker node.
  # param ret 1 for yes, 0 for no
  # return 0 when success, -1 when failure happens
  fun mx_kv_store_is_worker_node = MXKVStoreIsWorkerNode(ret : Int32*) : Int32

  # return whether or not this process is a server node.
  # param ret 1 for yes, 0 for no
  # return 0 when success, -1 when failure happens
  fun mx_kv_store_is_server_node = MXKVStoreIsServerNode(ret : Int32*) : Int32
  # return whether or not this process is a scheduler node.
  # param ret 1 for yes, 0 for no
  # return 0 when success, -1 when failure happens
  fun mx_kv_store_is_scheduler_node = MXKVStoreIsSchedulerNode(ret : Int32*) : Int32
  # global barrier among all worker machines
  #
  # param handle handle to the KVStore
  # return 0 when success, -1 when failure happens
  fun mx_kv_store_barrier = MXKVStoreBarrier(handle : KVStoreHandle) : Int32
  # the prototype of a server controller
  # param head the head of the command
  # param body the body of the command
  # param controller_handle helper handle for implementing controller
  type MXKVStoreServerController = Int32, UInt8*, Void* -> Void

  # return Run as server (or scheduler)
  #
  # param handle handle to the KVStore
  # param controller the user-defined server controller
  # param controller_handle helper handle for implementing controller
  # return 0 when success, -1 when failure happens
  fun mx_kv_store_run_server = MXKVStoreRunServer(handle : KVStoreHandle, controller : MXKVStoreServerController, controller_handle : Void*) : Int32
  # return Send a command to all server nodes
  #
  # param handle handle to the KVStore
  # param cmd_id the head of the command
  # param cmd_body the body of the command
  # return 0 when success, -1 when failure happens
  fun mx_kv_store_send_command_to_servers = MXKVStoreSendCommmandToServers(handle : KVStoreHandle, cmd_id : Int32, cmd_body : UInt8*) : Int32
  # Create a RecordIO writer object
  # param uri path to file
  # param out handle pointer to the created object
  # return 0 when success, -1 when failure happens
  fun mx_record_io_writer_create = MXRecordIOWriterCreate(uri : UInt8*, out_ : RecordIOHandle*) : Int32
  # Delete a RecordIO writer object
  # param handle handle to RecordIO object
  # return 0 when success, -1 when failure happens
  fun mx_record_io_writer_free = MXRecordIOWriterFree(handle : RecordIOHandle) : Int32
  # Write a record to a RecordIO object
  # param handle handle to RecordIO object
  # param buf buffer to write
  # param size size of buffer
  # return 0 when success, -1 when failure happens
  fun mx_record_io_writer_write_record = MXRecordIOWriterWriteRecord(handle : RecordIOHandle*,
                                                                     buf : UInt8*,
                                                                     size : LibC::SizeT) : Int32
  # Create a RecordIO reader object
  # param uri path to file
  # param out handle pointer to the created object
  # return 0 when success, -1 when failure happens
  fun mx_record_io_reader_create = MXRecordIOReaderCreate(uri : UInt8*, out_ : RecordIOHandle*) : Int32
  # Delete a RecordIO reader object
  # param handle handle to RecordIO object
  # return 0 when success, -1 when failure happens
  fun mx_record_io_reader_free = MXRecordIOReaderFree(handle : RecordIOHandle*) : Int32
  # Write a record to a RecordIO object
  # param handle handle to RecordIO object
  # param buf pointer to return buffer
  # param size point to size of buffer
  # return 0 when success, -1 when failure happens
  fun mx_record_io_reader_read_record = MXRecordIOReaderReadRecord(handle : RecordIOHandle*, buf : UInt8**, size : LibC::SizeT*) : Int32
  # Create a MXRtc object
  fun mx_rtc_create = MXRtcCreate(name : UInt8*, num_input : MXUInt, num_output : MXUInt,
                                  input_names : UInt8**, output_names : UInt8**,
                                  inputs : NDArrayHandle*, outputs : NDArrayHandle*,
                                  kernel : UInt8*, out_ : RtcHandle*) : Int32
  # Run cuda kernel
  fun mx_rtc_push = MXRtcPush(handle : RtcHandle, num_input : MXUInt, num_output : MXUInt,
                              inputs : NDArrayHandle*, outputs : NDArrayHandle*,
                              gridDimX : MXUInt, gridDimY : MXUInt, gridDimZ : MXUInt,
                              blockDimX : MXUInt, blockDimY : MXUInt, blockDimZ : MXUInt) : Int32
  # Delete a MXRtc object
  fun mx_rtc_free = MXRtcFree(handle : RtcHandle) : Int32
  fun mx_optimizer_find_creator = MXOptimizerFindCreator(key : UInt8*, out_ : OptimizerCreator*) : Int32
  fun mx_optimizer_create_optimizer = MXOptimizerCreateOptimizer(creator : OptimizerCreator, num_param : MXUInt,
                                                                 keys : UInt8**, vals : UInt8**,
                                                                 out_ : OptimizerHandle*) : Int32
  fun mx_optimizer_free = MXOptimizerFree(handle : OptimizerHandle) : Int32
  fun mx_optimizer_update = MXOptimizerUpdate(handle : OptimizerHandle, index : Int32,
                                              weight : NDArrayHandle, grad : NDArrayHandle,
                                              lr : MXFloat, wd : MXFloat) : Int32
end
