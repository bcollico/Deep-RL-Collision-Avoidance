??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
:
Minimum
x"T
y"T
z"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8??
w
layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namelayer1/kernel
p
!layer1/kernel/Read/ReadVariableOpReadVariableOplayer1/kernel*
_output_shapes
:	?*
dtype0
o
layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namelayer1/bias
h
layer1/bias/Read/ReadVariableOpReadVariableOplayer1/bias*
_output_shapes	
:?*
dtype0
w
layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*
shared_namelayer2/kernel
p
!layer2/kernel/Read/ReadVariableOpReadVariableOplayer2/kernel*
_output_shapes
:	?d*
dtype0
n
layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namelayer2/bias
g
layer2/bias/Read/ReadVariableOpReadVariableOplayer2/bias*
_output_shapes
:d*
dtype0
v
layer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_namelayer3/kernel
o
!layer3/kernel/Read/ReadVariableOpReadVariableOplayer3/kernel*
_output_shapes

:dd*
dtype0
n
layer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namelayer3/bias
g
layer3/bias/Read/ReadVariableOpReadVariableOplayer3/bias*
_output_shapes
:d*
dtype0
?
final_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*#
shared_namefinal_layer/kernel
y
&final_layer/kernel/Read/ReadVariableOpReadVariableOpfinal_layer/kernel*
_output_shapes

:d*
dtype0
x
final_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namefinal_layer/bias
q
$final_layer/bias/Read/ReadVariableOpReadVariableOpfinal_layer/bias*
_output_shapes
:*
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
d
momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
momentum
]
momentum/Read/ReadVariableOpReadVariableOpmomentum*
_output_shapes
: *
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
SGD/layer1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*+
shared_nameSGD/layer1/kernel/momentum
?
.SGD/layer1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/layer1/kernel/momentum*
_output_shapes
:	?*
dtype0
?
SGD/layer1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameSGD/layer1/bias/momentum
?
,SGD/layer1/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/layer1/bias/momentum*
_output_shapes	
:?*
dtype0
?
SGD/layer2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*+
shared_nameSGD/layer2/kernel/momentum
?
.SGD/layer2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/layer2/kernel/momentum*
_output_shapes
:	?d*
dtype0
?
SGD/layer2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*)
shared_nameSGD/layer2/bias/momentum
?
,SGD/layer2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/layer2/bias/momentum*
_output_shapes
:d*
dtype0
?
SGD/layer3/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*+
shared_nameSGD/layer3/kernel/momentum
?
.SGD/layer3/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/layer3/kernel/momentum*
_output_shapes

:dd*
dtype0
?
SGD/layer3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*)
shared_nameSGD/layer3/bias/momentum
?
,SGD/layer3/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/layer3/bias/momentum*
_output_shapes
:d*
dtype0
?
SGD/final_layer/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*0
shared_name!SGD/final_layer/kernel/momentum
?
3SGD/final_layer/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/final_layer/kernel/momentum*
_output_shapes

:d*
dtype0
?
SGD/final_layer/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameSGD/final_layer/bias/momentum
?
1SGD/final_layer/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/final_layer/bias/momentum*
_output_shapes
:*
dtype0

NoOpNoOp
?&
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?%
value?%B?% B?%
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
R
$	variables
%regularization_losses
&trainable_variables
'	keras_api
?
	(decay
)learning_rate
*momentum
+itermomentumOmomentumPmomentumQmomentumRmomentumSmomentumTmomentumUmomentumV
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
 
?
,non_trainable_variables

-layers
	variables
.layer_metrics
/metrics
trainable_variables
	regularization_losses
0layer_regularization_losses
 
YW
VARIABLE_VALUElayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
1non_trainable_variables

2layers
	variables
3layer_metrics
4metrics
regularization_losses
trainable_variables
5layer_regularization_losses
YW
VARIABLE_VALUElayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
6non_trainable_variables

7layers
	variables
8layer_metrics
9metrics
regularization_losses
trainable_variables
:layer_regularization_losses
YW
VARIABLE_VALUElayer3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
;non_trainable_variables

<layers
	variables
=layer_metrics
>metrics
regularization_losses
trainable_variables
?layer_regularization_losses
^\
VARIABLE_VALUEfinal_layer/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEfinal_layer/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
@non_trainable_variables

Alayers
 	variables
Blayer_metrics
Cmetrics
!regularization_losses
"trainable_variables
Dlayer_regularization_losses
 
 
 
?
Enon_trainable_variables

Flayers
$	variables
Glayer_metrics
Hmetrics
%regularization_losses
&trainable_variables
Ilayer_regularization_losses
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4
 

J0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ktotal
	Lcount
M	variables
N	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

M	variables
??
VARIABLE_VALUESGD/layer1/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/layer1/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/layer2/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/layer2/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/layer3/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/layer3/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/final_layer/kernel/momentumYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/final_layer/bias/momentumWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1layer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biasfinal_layer/kernelfinal_layer/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_12119
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!layer1/kernel/Read/ReadVariableOplayer1/bias/Read/ReadVariableOp!layer2/kernel/Read/ReadVariableOplayer2/bias/Read/ReadVariableOp!layer3/kernel/Read/ReadVariableOplayer3/bias/Read/ReadVariableOp&final_layer/kernel/Read/ReadVariableOp$final_layer/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOpSGD/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.SGD/layer1/kernel/momentum/Read/ReadVariableOp,SGD/layer1/bias/momentum/Read/ReadVariableOp.SGD/layer2/kernel/momentum/Read/ReadVariableOp,SGD/layer2/bias/momentum/Read/ReadVariableOp.SGD/layer3/kernel/momentum/Read/ReadVariableOp,SGD/layer3/bias/momentum/Read/ReadVariableOp3SGD/final_layer/kernel/momentum/Read/ReadVariableOp1SGD/final_layer/bias/momentum/Read/ReadVariableOpConst*#
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_12492
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biasfinal_layer/kernelfinal_layer/biasdecaylearning_ratemomentumSGD/itertotalcountSGD/layer1/kernel/momentumSGD/layer1/bias/momentumSGD/layer2/kernel/momentumSGD/layer2/bias/momentumSGD/layer3/kernel/momentumSGD/layer3/bias/momentumSGD/final_layer/kernel/momentumSGD/final_layer/bias/momentum*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_12568??
?	
?
A__inference_layer2_layer_call_and_return_conditional_losses_12292

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
A__inference_layer2_layer_call_and_return_conditional_losses_11796

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_12240

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_119952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
[
?__inference_clip_layer_call_and_return_conditional_losses_11888

inputs
identityw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimuminputs clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valuee
IdentityIdentityclip_by_value:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_12403>
:final_layer_kernel_regularizer_abs_readvariableop_resource
identity??1final_layer/kernel/Regularizer/Abs/ReadVariableOp?4final_layer/kernel/Regularizer/Square/ReadVariableOp?
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$final_layer/kernel/Regularizer/Const?
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp:final_layer_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:d*
dtype023
1final_layer/kernel/Regularizer/Abs/ReadVariableOp?
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"final_layer/kernel/Regularizer/Abs?
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_1?
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/Sum?
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22&
$final_layer/kernel/Regularizer/mul/x?
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/mul?
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/add?
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:final_layer_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:d*
dtype026
4final_layer/kernel/Regularizer/Square/ReadVariableOp?
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2'
%final_layer/kernel/Regularizer/Square?
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_2?
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/Sum_1?
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???32(
&final_layer/kernel/Regularizer/mul_1/x?
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/mul_1?
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/add_1?
IdentityIdentity(final_layer/kernel/Regularizer/add_1:z:02^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1final_layer/kernel/Regularizer/Abs/ReadVariableOp1final_layer/kernel/Regularizer/Abs/ReadVariableOp2l
4final_layer/kernel/Regularizer/Square/ReadVariableOp4final_layer/kernel/Regularizer/Square/ReadVariableOp
?1
?
E__inference_sequential_layer_call_and_return_conditional_losses_11952
input_1
layer1_11915
layer1_11917
layer2_11920
layer2_11922
layer3_11925
layer3_11927
final_layer_11930
final_layer_11932
identity??#final_layer/StatefulPartitionedCall?1final_layer/kernel/Regularizer/Abs/ReadVariableOp?4final_layer/kernel/Regularizer/Square/ReadVariableOp?layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer3/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinput_1layer1_11915layer1_11917*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_117692 
layer1/StatefulPartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_11920layer2_11922*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_117962 
layer2/StatefulPartitionedCall?
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_11925layer3_11927*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_118232 
layer3/StatefulPartitionedCall?
#final_layer/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0final_layer_11930final_layer_11932*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_final_layer_layer_call_and_return_conditional_losses_118642%
#final_layer/StatefulPartitionedCall?
clip/PartitionedCallPartitionedCall,final_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_clip_layer_call_and_return_conditional_losses_118882
clip/PartitionedCall?
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$final_layer/kernel/Regularizer/Const?
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpfinal_layer_11930*
_output_shapes

:d*
dtype023
1final_layer/kernel/Regularizer/Abs/ReadVariableOp?
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"final_layer/kernel/Regularizer/Abs?
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_1?
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/Sum?
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22&
$final_layer/kernel/Regularizer/mul/x?
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/mul?
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/add?
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfinal_layer_11930*
_output_shapes

:d*
dtype026
4final_layer/kernel/Regularizer/Square/ReadVariableOp?
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2'
%final_layer/kernel/Regularizer/Square?
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_2?
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/Sum_1?
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???32(
&final_layer/kernel/Regularizer/mul_1/x?
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/mul_1?
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/add_1?
IdentityIdentityclip/PartitionedCall:output:0$^final_layer/StatefulPartitionedCall2^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2J
#final_layer/StatefulPartitionedCall#final_layer/StatefulPartitionedCall2f
1final_layer/kernel/Regularizer/Abs/ReadVariableOp1final_layer/kernel/Regularizer/Abs/ReadVariableOp2l
4final_layer/kernel/Regularizer/Square/ReadVariableOp4final_layer/kernel/Regularizer/Square/ReadVariableOp2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
#__inference_signature_wrapper_12119
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_117542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?1
?
E__inference_sequential_layer_call_and_return_conditional_losses_11912
input_1
layer1_11780
layer1_11782
layer2_11807
layer2_11809
layer3_11834
layer3_11836
final_layer_11875
final_layer_11877
identity??#final_layer/StatefulPartitionedCall?1final_layer/kernel/Regularizer/Abs/ReadVariableOp?4final_layer/kernel/Regularizer/Square/ReadVariableOp?layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer3/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinput_1layer1_11780layer1_11782*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_117692 
layer1/StatefulPartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_11807layer2_11809*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_117962 
layer2/StatefulPartitionedCall?
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_11834layer3_11836*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_118232 
layer3/StatefulPartitionedCall?
#final_layer/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0final_layer_11875final_layer_11877*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_final_layer_layer_call_and_return_conditional_losses_118642%
#final_layer/StatefulPartitionedCall?
clip/PartitionedCallPartitionedCall,final_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_clip_layer_call_and_return_conditional_losses_118882
clip/PartitionedCall?
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$final_layer/kernel/Regularizer/Const?
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpfinal_layer_11875*
_output_shapes

:d*
dtype023
1final_layer/kernel/Regularizer/Abs/ReadVariableOp?
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"final_layer/kernel/Regularizer/Abs?
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_1?
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/Sum?
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22&
$final_layer/kernel/Regularizer/mul/x?
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/mul?
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/add?
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfinal_layer_11875*
_output_shapes

:d*
dtype026
4final_layer/kernel/Regularizer/Square/ReadVariableOp?
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2'
%final_layer/kernel/Regularizer/Square?
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_2?
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/Sum_1?
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???32(
&final_layer/kernel/Regularizer/mul_1/x?
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/mul_1?
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/add_1?
IdentityIdentityclip/PartitionedCall:output:0$^final_layer/StatefulPartitionedCall2^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2J
#final_layer/StatefulPartitionedCall#final_layer/StatefulPartitionedCall2f
1final_layer/kernel/Regularizer/Abs/ReadVariableOp1final_layer/kernel/Regularizer/Abs/ReadVariableOp2l
4final_layer/kernel/Regularizer/Square/ReadVariableOp4final_layer/kernel/Regularizer/Square/ReadVariableOp2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
*__inference_sequential_layer_call_fn_12075
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_120562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
A__inference_layer1_layer_call_and_return_conditional_losses_11769

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
{
&__inference_layer1_layer_call_fn_12281

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_117692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
@
$__inference_clip_layer_call_fn_12383

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_clip_layer_call_and_return_conditional_losses_118882
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_final_layer_layer_call_fn_12370

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_final_layer_layer_call_and_return_conditional_losses_118642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?3
?
 __inference__wrapped_model_11754
input_14
0sequential_layer1_matmul_readvariableop_resource5
1sequential_layer1_biasadd_readvariableop_resource4
0sequential_layer2_matmul_readvariableop_resource5
1sequential_layer2_biasadd_readvariableop_resource4
0sequential_layer3_matmul_readvariableop_resource5
1sequential_layer3_biasadd_readvariableop_resource9
5sequential_final_layer_matmul_readvariableop_resource:
6sequential_final_layer_biasadd_readvariableop_resource
identity??-sequential/final_layer/BiasAdd/ReadVariableOp?,sequential/final_layer/MatMul/ReadVariableOp?(sequential/layer1/BiasAdd/ReadVariableOp?'sequential/layer1/MatMul/ReadVariableOp?(sequential/layer2/BiasAdd/ReadVariableOp?'sequential/layer2/MatMul/ReadVariableOp?(sequential/layer3/BiasAdd/ReadVariableOp?'sequential/layer3/MatMul/ReadVariableOp?
'sequential/layer1/MatMul/ReadVariableOpReadVariableOp0sequential_layer1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'sequential/layer1/MatMul/ReadVariableOp?
sequential/layer1/MatMulMatMulinput_1/sequential/layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/layer1/MatMul?
(sequential/layer1/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(sequential/layer1/BiasAdd/ReadVariableOp?
sequential/layer1/BiasAddBiasAdd"sequential/layer1/MatMul:product:00sequential/layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/layer1/BiasAdd?
sequential/layer1/ReluRelu"sequential/layer1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/layer1/Relu?
'sequential/layer2/MatMul/ReadVariableOpReadVariableOp0sequential_layer2_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02)
'sequential/layer2/MatMul/ReadVariableOp?
sequential/layer2/MatMulMatMul$sequential/layer1/Relu:activations:0/sequential/layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential/layer2/MatMul?
(sequential/layer2/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02*
(sequential/layer2/BiasAdd/ReadVariableOp?
sequential/layer2/BiasAddBiasAdd"sequential/layer2/MatMul:product:00sequential/layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential/layer2/BiasAdd?
sequential/layer2/ReluRelu"sequential/layer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential/layer2/Relu?
'sequential/layer3/MatMul/ReadVariableOpReadVariableOp0sequential_layer3_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02)
'sequential/layer3/MatMul/ReadVariableOp?
sequential/layer3/MatMulMatMul$sequential/layer2/Relu:activations:0/sequential/layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential/layer3/MatMul?
(sequential/layer3/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02*
(sequential/layer3/BiasAdd/ReadVariableOp?
sequential/layer3/BiasAddBiasAdd"sequential/layer3/MatMul:product:00sequential/layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential/layer3/BiasAdd?
sequential/layer3/ReluRelu"sequential/layer3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential/layer3/Relu?
,sequential/final_layer/MatMul/ReadVariableOpReadVariableOp5sequential_final_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential/final_layer/MatMul/ReadVariableOp?
sequential/final_layer/MatMulMatMul$sequential/layer3/Relu:activations:04sequential/final_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/final_layer/MatMul?
-sequential/final_layer/BiasAdd/ReadVariableOpReadVariableOp6sequential_final_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/final_layer/BiasAdd/ReadVariableOp?
sequential/final_layer/BiasAddBiasAdd'sequential/final_layer/MatMul:product:05sequential/final_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential/final_layer/BiasAdd?
'sequential/clip/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2)
'sequential/clip/clip_by_value/Minimum/y?
%sequential/clip/clip_by_value/MinimumMinimum'sequential/final_layer/BiasAdd:output:00sequential/clip/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2'
%sequential/clip/clip_by_value/Minimum?
sequential/clip/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential/clip/clip_by_value/y?
sequential/clip/clip_by_valueMaximum)sequential/clip/clip_by_value/Minimum:z:0(sequential/clip/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
sequential/clip/clip_by_value?
IdentityIdentity!sequential/clip/clip_by_value:z:0.^sequential/final_layer/BiasAdd/ReadVariableOp-^sequential/final_layer/MatMul/ReadVariableOp)^sequential/layer1/BiasAdd/ReadVariableOp(^sequential/layer1/MatMul/ReadVariableOp)^sequential/layer2/BiasAdd/ReadVariableOp(^sequential/layer2/MatMul/ReadVariableOp)^sequential/layer3/BiasAdd/ReadVariableOp(^sequential/layer3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2^
-sequential/final_layer/BiasAdd/ReadVariableOp-sequential/final_layer/BiasAdd/ReadVariableOp2\
,sequential/final_layer/MatMul/ReadVariableOp,sequential/final_layer/MatMul/ReadVariableOp2T
(sequential/layer1/BiasAdd/ReadVariableOp(sequential/layer1/BiasAdd/ReadVariableOp2R
'sequential/layer1/MatMul/ReadVariableOp'sequential/layer1/MatMul/ReadVariableOp2T
(sequential/layer2/BiasAdd/ReadVariableOp(sequential/layer2/BiasAdd/ReadVariableOp2R
'sequential/layer2/MatMul/ReadVariableOp'sequential/layer2/MatMul/ReadVariableOp2T
(sequential/layer3/BiasAdd/ReadVariableOp(sequential/layer3/BiasAdd/ReadVariableOp2R
'sequential/layer3/MatMul/ReadVariableOp'sequential/layer3/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?5
?	
__inference__traced_save_12492
file_prefix,
(savev2_layer1_kernel_read_readvariableop*
&savev2_layer1_bias_read_readvariableop,
(savev2_layer2_kernel_read_readvariableop*
&savev2_layer2_bias_read_readvariableop,
(savev2_layer3_kernel_read_readvariableop*
&savev2_layer3_bias_read_readvariableop1
-savev2_final_layer_kernel_read_readvariableop/
+savev2_final_layer_bias_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop9
5savev2_sgd_layer1_kernel_momentum_read_readvariableop7
3savev2_sgd_layer1_bias_momentum_read_readvariableop9
5savev2_sgd_layer2_kernel_momentum_read_readvariableop7
3savev2_sgd_layer2_bias_momentum_read_readvariableop9
5savev2_sgd_layer3_kernel_momentum_read_readvariableop7
3savev2_sgd_layer3_bias_momentum_read_readvariableop>
:savev2_sgd_final_layer_kernel_momentum_read_readvariableop<
8savev2_sgd_final_layer_bias_momentum_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_layer2_kernel_read_readvariableop&savev2_layer2_bias_read_readvariableop(savev2_layer3_kernel_read_readvariableop&savev2_layer3_bias_read_readvariableop-savev2_final_layer_kernel_read_readvariableop+savev2_final_layer_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_sgd_layer1_kernel_momentum_read_readvariableop3savev2_sgd_layer1_bias_momentum_read_readvariableop5savev2_sgd_layer2_kernel_momentum_read_readvariableop3savev2_sgd_layer2_bias_momentum_read_readvariableop5savev2_sgd_layer3_kernel_momentum_read_readvariableop3savev2_sgd_layer3_bias_momentum_read_readvariableop:savev2_sgd_final_layer_kernel_momentum_read_readvariableop8savev2_sgd_final_layer_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?:?:	?d:d:dd:d:d:: : : : : : :	?:?:	?d:d:dd:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: 
?	
?
A__inference_layer3_layer_call_and_return_conditional_losses_11823

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
[
?__inference_clip_layer_call_and_return_conditional_losses_12378

inputs
identityw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimuminputs clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valuee
IdentityIdentityclip_by_value:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_12261

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_120562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
F__inference_final_layer_layer_call_and_return_conditional_losses_11864

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1final_layer/kernel/Regularizer/Abs/ReadVariableOp?4final_layer/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$final_layer/kernel/Regularizer/Const?
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1final_layer/kernel/Regularizer/Abs/ReadVariableOp?
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"final_layer/kernel/Regularizer/Abs?
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_1?
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/Sum?
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22&
$final_layer/kernel/Regularizer/mul/x?
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/mul?
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/add?
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype026
4final_layer/kernel/Regularizer/Square/ReadVariableOp?
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2'
%final_layer/kernel/Regularizer/Square?
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_2?
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/Sum_1?
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???32(
&final_layer/kernel/Regularizer/mul_1/x?
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/mul_1?
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/add_1?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1final_layer/kernel/Regularizer/Abs/ReadVariableOp1final_layer/kernel/Regularizer/Abs/ReadVariableOp2l
4final_layer/kernel/Regularizer/Square/ReadVariableOp4final_layer/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?1
?
E__inference_sequential_layer_call_and_return_conditional_losses_11995

inputs
layer1_11958
layer1_11960
layer2_11963
layer2_11965
layer3_11968
layer3_11970
final_layer_11973
final_layer_11975
identity??#final_layer/StatefulPartitionedCall?1final_layer/kernel/Regularizer/Abs/ReadVariableOp?4final_layer/kernel/Regularizer/Square/ReadVariableOp?layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer3/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_11958layer1_11960*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_117692 
layer1/StatefulPartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_11963layer2_11965*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_117962 
layer2/StatefulPartitionedCall?
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_11968layer3_11970*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_118232 
layer3/StatefulPartitionedCall?
#final_layer/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0final_layer_11973final_layer_11975*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_final_layer_layer_call_and_return_conditional_losses_118642%
#final_layer/StatefulPartitionedCall?
clip/PartitionedCallPartitionedCall,final_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_clip_layer_call_and_return_conditional_losses_118882
clip/PartitionedCall?
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$final_layer/kernel/Regularizer/Const?
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpfinal_layer_11973*
_output_shapes

:d*
dtype023
1final_layer/kernel/Regularizer/Abs/ReadVariableOp?
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"final_layer/kernel/Regularizer/Abs?
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_1?
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/Sum?
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22&
$final_layer/kernel/Regularizer/mul/x?
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/mul?
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/add?
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfinal_layer_11973*
_output_shapes

:d*
dtype026
4final_layer/kernel/Regularizer/Square/ReadVariableOp?
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2'
%final_layer/kernel/Regularizer/Square?
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_2?
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/Sum_1?
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???32(
&final_layer/kernel/Regularizer/mul_1/x?
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/mul_1?
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/add_1?
IdentityIdentityclip/PartitionedCall:output:0$^final_layer/StatefulPartitionedCall2^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2J
#final_layer/StatefulPartitionedCall#final_layer/StatefulPartitionedCall2f
1final_layer/kernel/Regularizer/Abs/ReadVariableOp1final_layer/kernel/Regularizer/Abs/ReadVariableOp2l
4final_layer/kernel/Regularizer/Square/ReadVariableOp4final_layer/kernel/Regularizer/Square/ReadVariableOp2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?C
?
E__inference_sequential_layer_call_and_return_conditional_losses_12169

inputs)
%layer1_matmul_readvariableop_resource*
&layer1_biasadd_readvariableop_resource)
%layer2_matmul_readvariableop_resource*
&layer2_biasadd_readvariableop_resource)
%layer3_matmul_readvariableop_resource*
&layer3_biasadd_readvariableop_resource.
*final_layer_matmul_readvariableop_resource/
+final_layer_biasadd_readvariableop_resource
identity??"final_layer/BiasAdd/ReadVariableOp?!final_layer/MatMul/ReadVariableOp?1final_layer/kernel/Regularizer/Abs/ReadVariableOp?4final_layer/kernel/Regularizer/Square/ReadVariableOp?layer1/BiasAdd/ReadVariableOp?layer1/MatMul/ReadVariableOp?layer2/BiasAdd/ReadVariableOp?layer2/MatMul/ReadVariableOp?layer3/BiasAdd/ReadVariableOp?layer3/MatMul/ReadVariableOp?
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
layer1/MatMul/ReadVariableOp?
layer1/MatMulMatMulinputs$layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer1/MatMul?
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer1/BiasAdd/ReadVariableOp?
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer1/BiasAddn
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
layer1/Relu?
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
layer2/MatMul/ReadVariableOp?
layer2/MatMulMatMullayer1/Relu:activations:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
layer2/MatMul?
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
layer2/BiasAdd/ReadVariableOp?
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
layer2/BiasAddm
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
layer2/Relu?
layer3/MatMul/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
layer3/MatMul/ReadVariableOp?
layer3/MatMulMatMullayer2/Relu:activations:0$layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
layer3/MatMul?
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
layer3/BiasAdd/ReadVariableOp?
layer3/BiasAddBiasAddlayer3/MatMul:product:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
layer3/BiasAddm
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
layer3/Relu?
!final_layer/MatMul/ReadVariableOpReadVariableOp*final_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02#
!final_layer/MatMul/ReadVariableOp?
final_layer/MatMulMatMullayer3/Relu:activations:0)final_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
final_layer/MatMul?
"final_layer/BiasAdd/ReadVariableOpReadVariableOp+final_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"final_layer/BiasAdd/ReadVariableOp?
final_layer/BiasAddBiasAddfinal_layer/MatMul:product:0*final_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
final_layer/BiasAdd?
clip/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
clip/clip_by_value/Minimum/y?
clip/clip_by_value/MinimumMinimumfinal_layer/BiasAdd:output:0%clip/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip/clip_by_value/Minimumq
clip/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip/clip_by_value/y?
clip/clip_by_valueMaximumclip/clip_by_value/Minimum:z:0clip/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip/clip_by_value?
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$final_layer/kernel/Regularizer/Const?
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp*final_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1final_layer/kernel/Regularizer/Abs/ReadVariableOp?
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"final_layer/kernel/Regularizer/Abs?
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_1?
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/Sum?
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22&
$final_layer/kernel/Regularizer/mul/x?
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/mul?
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/add?
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*final_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype026
4final_layer/kernel/Regularizer/Square/ReadVariableOp?
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2'
%final_layer/kernel/Regularizer/Square?
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_2?
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/Sum_1?
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???32(
&final_layer/kernel/Regularizer/mul_1/x?
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/mul_1?
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/add_1?
IdentityIdentityclip/clip_by_value:z:0#^final_layer/BiasAdd/ReadVariableOp"^final_layer/MatMul/ReadVariableOp2^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2H
"final_layer/BiasAdd/ReadVariableOp"final_layer/BiasAdd/ReadVariableOp2F
!final_layer/MatMul/ReadVariableOp!final_layer/MatMul/ReadVariableOp2f
1final_layer/kernel/Regularizer/Abs/ReadVariableOp1final_layer/kernel/Regularizer/Abs/ReadVariableOp2l
4final_layer/kernel/Regularizer/Square/ReadVariableOp4final_layer/kernel/Regularizer/Square/ReadVariableOp2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/MatMul/ReadVariableOplayer3/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_12014
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_119952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
{
&__inference_layer3_layer_call_fn_12321

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_118232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?1
?
E__inference_sequential_layer_call_and_return_conditional_losses_12056

inputs
layer1_12019
layer1_12021
layer2_12024
layer2_12026
layer3_12029
layer3_12031
final_layer_12034
final_layer_12036
identity??#final_layer/StatefulPartitionedCall?1final_layer/kernel/Regularizer/Abs/ReadVariableOp?4final_layer/kernel/Regularizer/Square/ReadVariableOp?layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer3/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_12019layer1_12021*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_117692 
layer1/StatefulPartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_12024layer2_12026*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_117962 
layer2/StatefulPartitionedCall?
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_12029layer3_12031*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_118232 
layer3/StatefulPartitionedCall?
#final_layer/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0final_layer_12034final_layer_12036*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_final_layer_layer_call_and_return_conditional_losses_118642%
#final_layer/StatefulPartitionedCall?
clip/PartitionedCallPartitionedCall,final_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_clip_layer_call_and_return_conditional_losses_118882
clip/PartitionedCall?
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$final_layer/kernel/Regularizer/Const?
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpfinal_layer_12034*
_output_shapes

:d*
dtype023
1final_layer/kernel/Regularizer/Abs/ReadVariableOp?
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"final_layer/kernel/Regularizer/Abs?
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_1?
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/Sum?
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22&
$final_layer/kernel/Regularizer/mul/x?
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/mul?
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/add?
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfinal_layer_12034*
_output_shapes

:d*
dtype026
4final_layer/kernel/Regularizer/Square/ReadVariableOp?
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2'
%final_layer/kernel/Regularizer/Square?
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_2?
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/Sum_1?
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???32(
&final_layer/kernel/Regularizer/mul_1/x?
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/mul_1?
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/add_1?
IdentityIdentityclip/PartitionedCall:output:0$^final_layer/StatefulPartitionedCall2^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2J
#final_layer/StatefulPartitionedCall#final_layer/StatefulPartitionedCall2f
1final_layer/kernel/Regularizer/Abs/ReadVariableOp1final_layer/kernel/Regularizer/Abs/ReadVariableOp2l
4final_layer/kernel/Regularizer/Square/ReadVariableOp4final_layer/kernel/Regularizer/Square/ReadVariableOp2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
A__inference_layer1_layer_call_and_return_conditional_losses_12272

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?C
?
E__inference_sequential_layer_call_and_return_conditional_losses_12219

inputs)
%layer1_matmul_readvariableop_resource*
&layer1_biasadd_readvariableop_resource)
%layer2_matmul_readvariableop_resource*
&layer2_biasadd_readvariableop_resource)
%layer3_matmul_readvariableop_resource*
&layer3_biasadd_readvariableop_resource.
*final_layer_matmul_readvariableop_resource/
+final_layer_biasadd_readvariableop_resource
identity??"final_layer/BiasAdd/ReadVariableOp?!final_layer/MatMul/ReadVariableOp?1final_layer/kernel/Regularizer/Abs/ReadVariableOp?4final_layer/kernel/Regularizer/Square/ReadVariableOp?layer1/BiasAdd/ReadVariableOp?layer1/MatMul/ReadVariableOp?layer2/BiasAdd/ReadVariableOp?layer2/MatMul/ReadVariableOp?layer3/BiasAdd/ReadVariableOp?layer3/MatMul/ReadVariableOp?
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
layer1/MatMul/ReadVariableOp?
layer1/MatMulMatMulinputs$layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer1/MatMul?
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer1/BiasAdd/ReadVariableOp?
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer1/BiasAddn
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
layer1/Relu?
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
layer2/MatMul/ReadVariableOp?
layer2/MatMulMatMullayer1/Relu:activations:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
layer2/MatMul?
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
layer2/BiasAdd/ReadVariableOp?
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
layer2/BiasAddm
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
layer2/Relu?
layer3/MatMul/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
layer3/MatMul/ReadVariableOp?
layer3/MatMulMatMullayer2/Relu:activations:0$layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
layer3/MatMul?
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
layer3/BiasAdd/ReadVariableOp?
layer3/BiasAddBiasAddlayer3/MatMul:product:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
layer3/BiasAddm
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
layer3/Relu?
!final_layer/MatMul/ReadVariableOpReadVariableOp*final_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02#
!final_layer/MatMul/ReadVariableOp?
final_layer/MatMulMatMullayer3/Relu:activations:0)final_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
final_layer/MatMul?
"final_layer/BiasAdd/ReadVariableOpReadVariableOp+final_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"final_layer/BiasAdd/ReadVariableOp?
final_layer/BiasAddBiasAddfinal_layer/MatMul:product:0*final_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
final_layer/BiasAdd?
clip/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
clip/clip_by_value/Minimum/y?
clip/clip_by_value/MinimumMinimumfinal_layer/BiasAdd:output:0%clip/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip/clip_by_value/Minimumq
clip/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip/clip_by_value/y?
clip/clip_by_valueMaximumclip/clip_by_value/Minimum:z:0clip/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip/clip_by_value?
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$final_layer/kernel/Regularizer/Const?
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp*final_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1final_layer/kernel/Regularizer/Abs/ReadVariableOp?
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"final_layer/kernel/Regularizer/Abs?
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_1?
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/Sum?
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22&
$final_layer/kernel/Regularizer/mul/x?
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/mul?
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/add?
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*final_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype026
4final_layer/kernel/Regularizer/Square/ReadVariableOp?
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2'
%final_layer/kernel/Regularizer/Square?
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_2?
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/Sum_1?
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???32(
&final_layer/kernel/Regularizer/mul_1/x?
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/mul_1?
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/add_1?
IdentityIdentityclip/clip_by_value:z:0#^final_layer/BiasAdd/ReadVariableOp"^final_layer/MatMul/ReadVariableOp2^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2H
"final_layer/BiasAdd/ReadVariableOp"final_layer/BiasAdd/ReadVariableOp2F
!final_layer/MatMul/ReadVariableOp!final_layer/MatMul/ReadVariableOp2f
1final_layer/kernel/Regularizer/Abs/ReadVariableOp1final_layer/kernel/Regularizer/Abs/ReadVariableOp2l
4final_layer/kernel/Regularizer/Square/ReadVariableOp4final_layer/kernel/Regularizer/Square/ReadVariableOp2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/MatMul/ReadVariableOplayer3/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
A__inference_layer3_layer_call_and_return_conditional_losses_12312

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
{
&__inference_layer2_layer_call_fn_12301

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_117962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?"
?
F__inference_final_layer_layer_call_and_return_conditional_losses_12361

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1final_layer/kernel/Regularizer/Abs/ReadVariableOp?4final_layer/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$final_layer/kernel/Regularizer/Const?
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1final_layer/kernel/Regularizer/Abs/ReadVariableOp?
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"final_layer/kernel/Regularizer/Abs?
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_1?
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/Sum?
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22&
$final_layer/kernel/Regularizer/mul/x?
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/mul?
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2$
"final_layer/kernel/Regularizer/add?
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype026
4final_layer/kernel/Regularizer/Square/ReadVariableOp?
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2'
%final_layer/kernel/Regularizer/Square?
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2(
&final_layer/kernel/Regularizer/Const_2?
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/Sum_1?
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???32(
&final_layer/kernel/Regularizer/mul_1/x?
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/mul_1?
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2&
$final_layer/kernel/Regularizer/add_1?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1final_layer/kernel/Regularizer/Abs/ReadVariableOp1final_layer/kernel/Regularizer/Abs/ReadVariableOp2l
4final_layer/kernel/Regularizer/Square/ReadVariableOp4final_layer/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?_
?
!__inference__traced_restore_12568
file_prefix"
assignvariableop_layer1_kernel"
assignvariableop_1_layer1_bias$
 assignvariableop_2_layer2_kernel"
assignvariableop_3_layer2_bias$
 assignvariableop_4_layer3_kernel"
assignvariableop_5_layer3_bias)
%assignvariableop_6_final_layer_kernel'
#assignvariableop_7_final_layer_bias
assignvariableop_8_decay$
 assignvariableop_9_learning_rate 
assignvariableop_10_momentum 
assignvariableop_11_sgd_iter
assignvariableop_12_total
assignvariableop_13_count2
.assignvariableop_14_sgd_layer1_kernel_momentum0
,assignvariableop_15_sgd_layer1_bias_momentum2
.assignvariableop_16_sgd_layer2_kernel_momentum0
,assignvariableop_17_sgd_layer2_bias_momentum2
.assignvariableop_18_sgd_layer3_kernel_momentum0
,assignvariableop_19_sgd_layer3_bias_momentum7
3assignvariableop_20_sgd_final_layer_kernel_momentum5
1assignvariableop_21_sgd_final_layer_bias_momentum
identity_23??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_layer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_layer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_layer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_layer3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_final_layer_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_final_layer_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_momentumIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_sgd_iterIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp.assignvariableop_14_sgd_layer1_kernel_momentumIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp,assignvariableop_15_sgd_layer1_bias_momentumIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp.assignvariableop_16_sgd_layer2_kernel_momentumIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp,assignvariableop_17_sgd_layer2_bias_momentumIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp.assignvariableop_18_sgd_layer3_kernel_momentumIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp,assignvariableop_19_sgd_layer3_bias_momentumIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp3assignvariableop_20_sgd_final_layer_kernel_momentumIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp1assignvariableop_21_sgd_final_layer_bias_momentumIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_219
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22?
Identity_23IdentityIdentity_22:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_23"#
identity_23Identity_23:output:0*m
_input_shapes\
Z: ::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????8
clip0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?+
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
W__call__
*X&call_and_return_all_conditional_losses
Y_default_save_signature"?(
_tf_keras_sequential?({"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}, {"class_name": "Dense", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "layer3", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "final_layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 9.99999993922529e-09, "l2": 1.0000000116860974e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Clip", "config": {"name": "clip", "trainable": true, "dtype": "float32"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 15}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}, {"class_name": "Dense", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "layer3", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "final_layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 9.99999993922529e-09, "l2": 1.0000000116860974e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Clip", "config": {"name": "clip", "trainable": true, "dtype": "float32"}}]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 9.999999747378752e-06, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 15}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15]}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
\__call__
*]&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer2", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 150}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150]}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
^__call__
*_&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer3", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
`__call__
*a&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "final_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "final_layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 9.99999993922529e-09, "l2": 1.0000000116860974e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?
$	variables
%regularization_losses
&trainable_variables
'	keras_api
b__call__
*c&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Clip", "name": "clip", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "clip", "trainable": true, "dtype": "float32"}}
?
	(decay
)learning_rate
*momentum
+itermomentumOmomentumPmomentumQmomentumRmomentumSmomentumTmomentumUmomentumV"
	optimizer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
'
d0"
trackable_list_wrapper
?
,non_trainable_variables

-layers
	variables
.layer_metrics
/metrics
trainable_variables
	regularization_losses
0layer_regularization_losses
W__call__
Y_default_save_signature
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
,
eserving_default"
signature_map
 :	?2layer1/kernel
:?2layer1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
1non_trainable_variables

2layers
	variables
3layer_metrics
4metrics
regularization_losses
trainable_variables
5layer_regularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
 :	?d2layer2/kernel
:d2layer2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
6non_trainable_variables

7layers
	variables
8layer_metrics
9metrics
regularization_losses
trainable_variables
:layer_regularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
:dd2layer3/kernel
:d2layer3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
;non_trainable_variables

<layers
	variables
=layer_metrics
>metrics
regularization_losses
trainable_variables
?layer_regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
$:"d2final_layer/kernel
:2final_layer/bias
.
0
1"
trackable_list_wrapper
'
d0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
@non_trainable_variables

Alayers
 	variables
Blayer_metrics
Cmetrics
!regularization_losses
"trainable_variables
Dlayer_regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Enon_trainable_variables

Flayers
$	variables
Glayer_metrics
Hmetrics
%regularization_losses
&trainable_variables
Ilayer_regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_dict_wrapper
'
J0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
d0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	Ktotal
	Lcount
M	variables
N	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
K0
L1"
trackable_list_wrapper
-
M	variables"
_generic_user_object
+:)	?2SGD/layer1/kernel/momentum
%:#?2SGD/layer1/bias/momentum
+:)	?d2SGD/layer2/kernel/momentum
$:"d2SGD/layer2/bias/momentum
*:(dd2SGD/layer3/kernel/momentum
$:"d2SGD/layer3/bias/momentum
/:-d2SGD/final_layer/kernel/momentum
):'2SGD/final_layer/bias/momentum
?2?
*__inference_sequential_layer_call_fn_12075
*__inference_sequential_layer_call_fn_12261
*__inference_sequential_layer_call_fn_12014
*__inference_sequential_layer_call_fn_12240?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_12169
E__inference_sequential_layer_call_and_return_conditional_losses_12219
E__inference_sequential_layer_call_and_return_conditional_losses_11952
E__inference_sequential_layer_call_and_return_conditional_losses_11912?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_11754?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
&__inference_layer1_layer_call_fn_12281?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_layer1_layer_call_and_return_conditional_losses_12272?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_layer2_layer_call_fn_12301?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_layer2_layer_call_and_return_conditional_losses_12292?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_layer3_layer_call_fn_12321?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_layer3_layer_call_and_return_conditional_losses_12312?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_final_layer_layer_call_fn_12370?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_final_layer_layer_call_and_return_conditional_losses_12361?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_clip_layer_call_fn_12383?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_clip_layer_call_and_return_conditional_losses_12378?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_12403?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
#__inference_signature_wrapper_12119input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_11754i0?-
&?#
!?
input_1?????????
? "+?(
&
clip?
clip??????????
?__inference_clip_layer_call_and_return_conditional_losses_12378X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? s
$__inference_clip_layer_call_fn_12383K/?,
%?"
 ?
inputs?????????
? "???????????
F__inference_final_layer_layer_call_and_return_conditional_losses_12361\/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? ~
+__inference_final_layer_layer_call_fn_12370O/?,
%?"
 ?
inputs?????????d
? "???????????
A__inference_layer1_layer_call_and_return_conditional_losses_12272]/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? z
&__inference_layer1_layer_call_fn_12281P/?,
%?"
 ?
inputs?????????
? "????????????
A__inference_layer2_layer_call_and_return_conditional_losses_12292]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? z
&__inference_layer2_layer_call_fn_12301P0?-
&?#
!?
inputs??????????
? "??????????d?
A__inference_layer3_layer_call_and_return_conditional_losses_12312\/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? y
&__inference_layer3_layer_call_fn_12321O/?,
%?"
 ?
inputs?????????d
? "??????????d:
__inference_loss_fn_0_12403?

? 
? "? ?
E__inference_sequential_layer_call_and_return_conditional_losses_11912k8?5
.?+
!?
input_1?????????
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_11952k8?5
.?+
!?
input_1?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_12169j7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_12219j7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
*__inference_sequential_layer_call_fn_12014^8?5
.?+
!?
input_1?????????
p

 
? "???????????
*__inference_sequential_layer_call_fn_12075^8?5
.?+
!?
input_1?????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_12240]7?4
-?*
 ?
inputs?????????
p

 
? "???????????
*__inference_sequential_layer_call_fn_12261]7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
#__inference_signature_wrapper_12119t;?8
? 
1?.
,
input_1!?
input_1?????????"+?(
&
clip?
clip?????????