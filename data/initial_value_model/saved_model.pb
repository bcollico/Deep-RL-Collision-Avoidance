þ
¢
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Á
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.7.02unknown8¤
w
layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namelayer1/kernel
p
!layer1/kernel/Read/ReadVariableOpReadVariableOplayer1/kernel*
_output_shapes
:	*
dtype0
o
layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer1/bias
h
layer1/bias/Read/ReadVariableOpReadVariableOplayer1/bias*
_output_shapes	
:*
dtype0
w
layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*
shared_namelayer2/kernel
p
!layer2/kernel/Read/ReadVariableOpReadVariableOplayer2/kernel*
_output_shapes
:	d*
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

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

SGD/layer1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_nameSGD/layer1/kernel/momentum

.SGD/layer1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/layer1/kernel/momentum*
_output_shapes
:	*
dtype0

SGD/layer1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameSGD/layer1/bias/momentum

,SGD/layer1/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/layer1/bias/momentum*
_output_shapes	
:*
dtype0

SGD/layer2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*+
shared_nameSGD/layer2/kernel/momentum

.SGD/layer2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/layer2/kernel/momentum*
_output_shapes
:	d*
dtype0

SGD/layer2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*)
shared_nameSGD/layer2/bias/momentum

,SGD/layer2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/layer2/bias/momentum*
_output_shapes
:d*
dtype0

SGD/layer3/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*+
shared_nameSGD/layer3/kernel/momentum

.SGD/layer3/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/layer3/kernel/momentum*
_output_shapes

:dd*
dtype0

SGD/layer3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*)
shared_nameSGD/layer3/bias/momentum

,SGD/layer3/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/layer3/bias/momentum*
_output_shapes
:d*
dtype0

SGD/final_layer/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*0
shared_name!SGD/final_layer/kernel/momentum

3SGD/final_layer/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/final_layer/kernel/momentum*
_output_shapes

:d*
dtype0

SGD/final_layer/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameSGD/final_layer/bias/momentum

1SGD/final_layer/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/final_layer/bias/momentum*
_output_shapes
:*
dtype0

NoOpNoOp
ð#
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*«#
value¡#B# B#

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
¶
	#decay
$learning_rate
%momentum
&itermomentumEmomentumFmomentumGmomentumHmomentumImomentumJmomentumKmomentumL
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
 
­
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
 
YW
VARIABLE_VALUElayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
YW
VARIABLE_VALUElayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
YW
VARIABLE_VALUElayer3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
^\
VARIABLE_VALUEfinal_layer/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEfinal_layer/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
 trainable_variables
!regularization_losses
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

@0
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
	Atotal
	Bcount
C	variables
D	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

A0
B1

C	variables

VARIABLE_VALUESGD/layer1/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/layer1/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/layer2/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/layer2/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/layer3/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/layer3/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/final_layer/kernel/momentumYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/final_layer/bias/momentumWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
¼
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1layer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biasfinal_layer/kernelfinal_layer/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_332844
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ý
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
GPU 2J 8 *(
f#R!
__inference__traced_save_333196
À
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_333272²
Ú*
Þ
F__inference_sequential_layer_call_and_return_conditional_losses_332563

inputs 
layer1_332477:	
layer1_332479:	 
layer2_332494:	d
layer2_332496:d
layer3_332511:dd
layer3_332513:d$
final_layer_332542:d 
final_layer_332544:
identity¢#final_layer/StatefulPartitionedCall¢1final_layer/kernel/Regularizer/Abs/ReadVariableOp¢4final_layer/kernel/Regularizer/Square/ReadVariableOp¢layer1/StatefulPartitionedCall¢layer2/StatefulPartitionedCall¢layer3/StatefulPartitionedCallé
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_332477layer1_332479*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_332476
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_332494layer2_332496*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_332493
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_332511layer3_332513*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_332510
#final_layer/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0final_layer_332542final_layer_332544*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_final_layer_layer_call_and_return_conditional_losses_332541i
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpfinal_layer_332542*
_output_shapes

:d*
dtype0
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       £
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: i
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+2¦
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfinal_layer_332542*
_output_shapes

:d*
dtype0
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¨
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: k
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¬
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
:  
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentity,final_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp$^final_layer/StatefulPartitionedCall2^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2J
#final_layer/StatefulPartitionedCall#final_layer/StatefulPartitionedCall2f
1final_layer/kernel/Regularizer/Abs/ReadVariableOp1final_layer/kernel/Regularizer/Abs/ReadVariableOp2l
4final_layer/kernel/Regularizer/Square/ReadVariableOp4final_layer/kernel/Regularizer/Square/ReadVariableOp2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
¾
+__inference_sequential_layer_call_fn_332724
input_1
unknown:	
	unknown_0:	
	unknown_1:	d
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:d
	unknown_6:
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_332684o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ð
ã
G__inference_final_layer_layer_call_and_return_conditional_losses_333087

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1final_layer/kernel/Regularizer/Abs/ReadVariableOp¢4final_layer/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       £
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: i
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+2¦
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¨
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: k
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¬
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
:  
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1final_layer/kernel/Regularizer/Abs/ReadVariableOp1final_layer/kernel/Regularizer/Abs/ReadVariableOp2l
4final_layer/kernel/Regularizer/Square/ReadVariableOp4final_layer/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ú*
Þ
F__inference_sequential_layer_call_and_return_conditional_losses_332684

inputs 
layer1_332648:	
layer1_332650:	 
layer2_332653:	d
layer2_332655:d
layer3_332658:dd
layer3_332660:d$
final_layer_332663:d 
final_layer_332665:
identity¢#final_layer/StatefulPartitionedCall¢1final_layer/kernel/Regularizer/Abs/ReadVariableOp¢4final_layer/kernel/Regularizer/Square/ReadVariableOp¢layer1/StatefulPartitionedCall¢layer2/StatefulPartitionedCall¢layer3/StatefulPartitionedCallé
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_332648layer1_332650*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_332476
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_332653layer2_332655*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_332493
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_332658layer3_332660*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_332510
#final_layer/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0final_layer_332663final_layer_332665*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_final_layer_layer_call_and_return_conditional_losses_332541i
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpfinal_layer_332663*
_output_shapes

:d*
dtype0
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       £
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: i
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+2¦
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfinal_layer_332663*
_output_shapes

:d*
dtype0
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¨
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: k
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¬
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
:  
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentity,final_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp$^final_layer/StatefulPartitionedCall2^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2J
#final_layer/StatefulPartitionedCall#final_layer/StatefulPartitionedCall2f
1final_layer/kernel/Regularizer/Abs/ReadVariableOp1final_layer/kernel/Regularizer/Abs/ReadVariableOp2l
4final_layer/kernel/Regularizer/Square/ReadVariableOp4final_layer/kernel/Regularizer/Square/ReadVariableOp2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾

'__inference_layer3_layer_call_fn_333027

inputs
unknown:dd
	unknown_0:d
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_332510o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
È	
¾
+__inference_sequential_layer_call_fn_332582
input_1
unknown:	
	unknown_0:	
	unknown_1:	d
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:d
	unknown_6:
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_332563o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¶4
ª	
__inference__traced_save_333196
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

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¸
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*á
value×BÔB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B ¬	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_layer2_kernel_read_readvariableop&savev2_layer2_bias_read_readvariableop(savev2_layer3_kernel_read_readvariableop&savev2_layer3_bias_read_readvariableop-savev2_final_layer_kernel_read_readvariableop+savev2_final_layer_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_sgd_layer1_kernel_momentum_read_readvariableop3savev2_sgd_layer1_bias_momentum_read_readvariableop5savev2_sgd_layer2_kernel_momentum_read_readvariableop3savev2_sgd_layer2_bias_momentum_read_readvariableop5savev2_sgd_layer3_kernel_momentum_read_readvariableop3savev2_sgd_layer3_bias_momentum_read_readvariableop:savev2_sgd_final_layer_kernel_momentum_read_readvariableop8savev2_sgd_final_layer_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*«
_input_shapes
: :	::	d:d:dd:d:d:: : : : : : :	::	d:d:dd:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:!

_output_shapes	
::%!

_output_shapes
:	d: 
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
:	:!

_output_shapes	
::%!

_output_shapes
:	d: 
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
Å	
½
+__inference_sequential_layer_call_fn_332865

inputs
unknown:	
	unknown_0:	
	unknown_1:	d
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:d
	unknown_6:
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_332563o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
·
$__inference_signature_wrapper_332844
input_1
unknown:	
	unknown_0:	
	unknown_1:	d
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:d
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_332458o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1


ó
B__inference_layer3_layer_call_and_return_conditional_losses_332510

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
­*
À
!__inference__wrapped_model_332458
input_1C
0sequential_layer1_matmul_readvariableop_resource:	@
1sequential_layer1_biasadd_readvariableop_resource:	C
0sequential_layer2_matmul_readvariableop_resource:	d?
1sequential_layer2_biasadd_readvariableop_resource:dB
0sequential_layer3_matmul_readvariableop_resource:dd?
1sequential_layer3_biasadd_readvariableop_resource:dG
5sequential_final_layer_matmul_readvariableop_resource:dD
6sequential_final_layer_biasadd_readvariableop_resource:
identity¢-sequential/final_layer/BiasAdd/ReadVariableOp¢,sequential/final_layer/MatMul/ReadVariableOp¢(sequential/layer1/BiasAdd/ReadVariableOp¢'sequential/layer1/MatMul/ReadVariableOp¢(sequential/layer2/BiasAdd/ReadVariableOp¢'sequential/layer2/MatMul/ReadVariableOp¢(sequential/layer3/BiasAdd/ReadVariableOp¢'sequential/layer3/MatMul/ReadVariableOp
'sequential/layer1/MatMul/ReadVariableOpReadVariableOp0sequential_layer1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
sequential/layer1/MatMulMatMulinput_1/sequential/layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(sequential/layer1/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
sequential/layer1/BiasAddBiasAdd"sequential/layer1/MatMul:product:00sequential/layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
sequential/layer1/ReluRelu"sequential/layer1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential/layer2/MatMul/ReadVariableOpReadVariableOp0sequential_layer2_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype0«
sequential/layer2/MatMulMatMul$sequential/layer1/Relu:activations:0/sequential/layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(sequential/layer2/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0¬
sequential/layer2/BiasAddBiasAdd"sequential/layer2/MatMul:product:00sequential/layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
sequential/layer2/ReluRelu"sequential/layer2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
'sequential/layer3/MatMul/ReadVariableOpReadVariableOp0sequential_layer3_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0«
sequential/layer3/MatMulMatMul$sequential/layer2/Relu:activations:0/sequential/layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(sequential/layer3/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0¬
sequential/layer3/BiasAddBiasAdd"sequential/layer3/MatMul:product:00sequential/layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
sequential/layer3/ReluRelu"sequential/layer3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
,sequential/final_layer/MatMul/ReadVariableOpReadVariableOp5sequential_final_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0µ
sequential/final_layer/MatMulMatMul$sequential/layer3/Relu:activations:04sequential/final_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential/final_layer/BiasAdd/ReadVariableOpReadVariableOp6sequential_final_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential/final_layer/BiasAddBiasAdd'sequential/final_layer/MatMul:product:05sequential/final_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity'sequential/final_layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
NoOpNoOp.^sequential/final_layer/BiasAdd/ReadVariableOp-^sequential/final_layer/MatMul/ReadVariableOp)^sequential/layer1/BiasAdd/ReadVariableOp(^sequential/layer1/MatMul/ReadVariableOp)^sequential/layer2/BiasAdd/ReadVariableOp(^sequential/layer2/MatMul/ReadVariableOp)^sequential/layer3/BiasAdd/ReadVariableOp(^sequential/layer3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2^
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
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Â

'__inference_layer1_layer_call_fn_332987

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_332476p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

,__inference_final_layer_layer_call_fn_333062

inputs
unknown:d
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_final_layer_layer_call_and_return_conditional_losses_332541o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ù7

F__inference_sequential_layer_call_and_return_conditional_losses_332932

inputs8
%layer1_matmul_readvariableop_resource:	5
&layer1_biasadd_readvariableop_resource:	8
%layer2_matmul_readvariableop_resource:	d4
&layer2_biasadd_readvariableop_resource:d7
%layer3_matmul_readvariableop_resource:dd4
&layer3_biasadd_readvariableop_resource:d<
*final_layer_matmul_readvariableop_resource:d9
+final_layer_biasadd_readvariableop_resource:
identity¢"final_layer/BiasAdd/ReadVariableOp¢!final_layer/MatMul/ReadVariableOp¢1final_layer/kernel/Regularizer/Abs/ReadVariableOp¢4final_layer/kernel/Regularizer/Square/ReadVariableOp¢layer1/BiasAdd/ReadVariableOp¢layer1/MatMul/ReadVariableOp¢layer2/BiasAdd/ReadVariableOp¢layer2/MatMul/ReadVariableOp¢layer3/BiasAdd/ReadVariableOp¢layer3/MatMul/ReadVariableOp
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0x
layer1/MatMulMatMulinputs$layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype0
layer2/MatMulMatMullayer1/Relu:activations:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
layer3/MatMul/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0
layer3/MatMulMatMullayer2/Relu:activations:0$layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
layer3/BiasAddBiasAddlayer3/MatMul:product:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!final_layer/MatMul/ReadVariableOpReadVariableOp*final_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
final_layer/MatMulMatMullayer3/Relu:activations:0)final_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"final_layer/BiasAdd/ReadVariableOpReadVariableOp+final_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
final_layer/BiasAddBiasAddfinal_layer/MatMul:product:0*final_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp*final_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       £
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: i
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+2¦
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*final_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¨
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: k
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¬
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
:  
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: k
IdentityIdentityfinal_layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
NoOpNoOp#^final_layer/BiasAdd/ReadVariableOp"^final_layer/MatMul/ReadVariableOp2^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2H
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ô
B__inference_layer2_layer_call_and_return_conditional_losses_333018

inputs1
matmul_readvariableop_resource:	d-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Z
£
"__inference__traced_restore_333272
file_prefix1
assignvariableop_layer1_kernel:	-
assignvariableop_1_layer1_bias:	3
 assignvariableop_2_layer2_kernel:	d,
assignvariableop_3_layer2_bias:d2
 assignvariableop_4_layer3_kernel:dd,
assignvariableop_5_layer3_bias:d7
%assignvariableop_6_final_layer_kernel:d1
#assignvariableop_7_final_layer_bias:"
assignvariableop_8_decay: *
 assignvariableop_9_learning_rate: &
assignvariableop_10_momentum: &
assignvariableop_11_sgd_iter:	 #
assignvariableop_12_total: #
assignvariableop_13_count: A
.assignvariableop_14_sgd_layer1_kernel_momentum:	;
,assignvariableop_15_sgd_layer1_bias_momentum:	A
.assignvariableop_16_sgd_layer2_kernel_momentum:	d:
,assignvariableop_17_sgd_layer2_bias_momentum:d@
.assignvariableop_18_sgd_layer3_kernel_momentum:dd:
,assignvariableop_19_sgd_layer3_bias_momentum:dE
3assignvariableop_20_sgd_final_layer_kernel_momentum:d?
1assignvariableop_21_sgd_final_layer_bias_momentum:
identity_23¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9»
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*á
value×BÔB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_layer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp assignvariableop_2_layer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_layer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp assignvariableop_4_layer3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_final_layer_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp#assignvariableop_7_final_layer_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_momentumIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_sgd_iterIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp.assignvariableop_14_sgd_layer1_kernel_momentumIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp,assignvariableop_15_sgd_layer1_bias_momentumIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp.assignvariableop_16_sgd_layer2_kernel_momentumIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp,assignvariableop_17_sgd_layer2_bias_momentumIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp.assignvariableop_18_sgd_layer3_kernel_momentumIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp,assignvariableop_19_sgd_layer3_bias_momentumIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_20AssignVariableOp3assignvariableop_20_sgd_final_layer_kernel_momentumIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_21AssignVariableOp1assignvariableop_21_sgd_final_layer_bias_momentumIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ³
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
:  
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
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
_user_specified_namefile_prefix
¡

õ
B__inference_layer1_layer_call_and_return_conditional_losses_332998

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î
è
__inference_loss_fn_0_333107L
:final_layer_kernel_regularizer_abs_readvariableop_resource:d
identity¢1final_layer/kernel/Regularizer/Abs/ReadVariableOp¢4final_layer/kernel/Regularizer/Square/ReadVariableOpi
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¬
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp:final_layer_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:d*
dtype0
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       £
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: i
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+2¦
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ¯
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:final_layer_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:d*
dtype0
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¨
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: k
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¬
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
:  
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
IdentityIdentity(final_layer/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ±
NoOpNoOp2^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1final_layer/kernel/Regularizer/Abs/ReadVariableOp1final_layer/kernel/Regularizer/Abs/ReadVariableOp2l
4final_layer/kernel/Regularizer/Square/ReadVariableOp4final_layer/kernel/Regularizer/Square/ReadVariableOp
Á

'__inference_layer2_layer_call_fn_333007

inputs
unknown:	d
	unknown_0:d
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_332493o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý*
ß
F__inference_sequential_layer_call_and_return_conditional_losses_332802
input_1 
layer1_332766:	
layer1_332768:	 
layer2_332771:	d
layer2_332773:d
layer3_332776:dd
layer3_332778:d$
final_layer_332781:d 
final_layer_332783:
identity¢#final_layer/StatefulPartitionedCall¢1final_layer/kernel/Regularizer/Abs/ReadVariableOp¢4final_layer/kernel/Regularizer/Square/ReadVariableOp¢layer1/StatefulPartitionedCall¢layer2/StatefulPartitionedCall¢layer3/StatefulPartitionedCallê
layer1/StatefulPartitionedCallStatefulPartitionedCallinput_1layer1_332766layer1_332768*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_332476
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_332771layer2_332773*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_332493
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_332776layer3_332778*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_332510
#final_layer/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0final_layer_332781final_layer_332783*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_final_layer_layer_call_and_return_conditional_losses_332541i
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpfinal_layer_332781*
_output_shapes

:d*
dtype0
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       £
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: i
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+2¦
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfinal_layer_332781*
_output_shapes

:d*
dtype0
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¨
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: k
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¬
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
:  
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentity,final_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp$^final_layer/StatefulPartitionedCall2^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2J
#final_layer/StatefulPartitionedCall#final_layer/StatefulPartitionedCall2f
1final_layer/kernel/Regularizer/Abs/ReadVariableOp1final_layer/kernel/Regularizer/Abs/ReadVariableOp2l
4final_layer/kernel/Regularizer/Square/ReadVariableOp4final_layer/kernel/Regularizer/Square/ReadVariableOp2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ý*
ß
F__inference_sequential_layer_call_and_return_conditional_losses_332763
input_1 
layer1_332727:	
layer1_332729:	 
layer2_332732:	d
layer2_332734:d
layer3_332737:dd
layer3_332739:d$
final_layer_332742:d 
final_layer_332744:
identity¢#final_layer/StatefulPartitionedCall¢1final_layer/kernel/Regularizer/Abs/ReadVariableOp¢4final_layer/kernel/Regularizer/Square/ReadVariableOp¢layer1/StatefulPartitionedCall¢layer2/StatefulPartitionedCall¢layer3/StatefulPartitionedCallê
layer1/StatefulPartitionedCallStatefulPartitionedCallinput_1layer1_332727layer1_332729*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_332476
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_332732layer2_332734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_332493
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_332737layer3_332739*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_332510
#final_layer/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0final_layer_332742final_layer_332744*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_final_layer_layer_call_and_return_conditional_losses_332541i
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpfinal_layer_332742*
_output_shapes

:d*
dtype0
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       £
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: i
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+2¦
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfinal_layer_332742*
_output_shapes

:d*
dtype0
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¨
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: k
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¬
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
:  
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentity,final_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp$^final_layer/StatefulPartitionedCall2^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2J
#final_layer/StatefulPartitionedCall#final_layer/StatefulPartitionedCall2f
1final_layer/kernel/Regularizer/Abs/ReadVariableOp1final_layer/kernel/Regularizer/Abs/ReadVariableOp2l
4final_layer/kernel/Regularizer/Square/ReadVariableOp4final_layer/kernel/Regularizer/Square/ReadVariableOp2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ð
ã
G__inference_final_layer_layer_call_and_return_conditional_losses_332541

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1final_layer/kernel/Regularizer/Abs/ReadVariableOp¢4final_layer/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       £
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: i
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+2¦
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¨
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: k
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¬
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
:  
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1final_layer/kernel/Regularizer/Abs/ReadVariableOp1final_layer/kernel/Regularizer/Abs/ReadVariableOp2l
4final_layer/kernel/Regularizer/Square/ReadVariableOp4final_layer/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs


ó
B__inference_layer3_layer_call_and_return_conditional_losses_333038

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs


ô
B__inference_layer2_layer_call_and_return_conditional_losses_332493

inputs1
matmul_readvariableop_resource:	d-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å	
½
+__inference_sequential_layer_call_fn_332886

inputs
unknown:	
	unknown_0:	
	unknown_1:	d
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:d
	unknown_6:
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_332684o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡

õ
B__inference_layer1_layer_call_and_return_conditional_losses_332476

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù7

F__inference_sequential_layer_call_and_return_conditional_losses_332978

inputs8
%layer1_matmul_readvariableop_resource:	5
&layer1_biasadd_readvariableop_resource:	8
%layer2_matmul_readvariableop_resource:	d4
&layer2_biasadd_readvariableop_resource:d7
%layer3_matmul_readvariableop_resource:dd4
&layer3_biasadd_readvariableop_resource:d<
*final_layer_matmul_readvariableop_resource:d9
+final_layer_biasadd_readvariableop_resource:
identity¢"final_layer/BiasAdd/ReadVariableOp¢!final_layer/MatMul/ReadVariableOp¢1final_layer/kernel/Regularizer/Abs/ReadVariableOp¢4final_layer/kernel/Regularizer/Square/ReadVariableOp¢layer1/BiasAdd/ReadVariableOp¢layer1/MatMul/ReadVariableOp¢layer2/BiasAdd/ReadVariableOp¢layer2/MatMul/ReadVariableOp¢layer3/BiasAdd/ReadVariableOp¢layer3/MatMul/ReadVariableOp
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0x
layer1/MatMulMatMulinputs$layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype0
layer2/MatMulMatMullayer1/Relu:activations:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
layer3/MatMul/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0
layer3/MatMulMatMullayer2/Relu:activations:0$layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
layer3/BiasAddBiasAddlayer3/MatMul:product:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!final_layer/MatMul/ReadVariableOpReadVariableOp*final_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
final_layer/MatMulMatMullayer3/Relu:activations:0)final_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"final_layer/BiasAdd/ReadVariableOpReadVariableOp+final_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
final_layer/BiasAddBiasAddfinal_layer/MatMul:product:0*final_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
$final_layer/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
1final_layer/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp*final_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
"final_layer/kernel/Regularizer/AbsAbs9final_layer/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       £
"final_layer/kernel/Regularizer/SumSum&final_layer/kernel/Regularizer/Abs:y:0/final_layer/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: i
$final_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+2¦
"final_layer/kernel/Regularizer/mulMul-final_layer/kernel/Regularizer/mul/x:output:0+final_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
"final_layer/kernel/Regularizer/addAddV2-final_layer/kernel/Regularizer/Const:output:0&final_layer/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
4final_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*final_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
%final_layer/kernel/Regularizer/SquareSquare<final_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&final_layer/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¨
$final_layer/kernel/Regularizer/Sum_1Sum)final_layer/kernel/Regularizer/Square:y:0/final_layer/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: k
&final_layer/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¬
$final_layer/kernel/Regularizer/mul_1Mul/final_layer/kernel/Regularizer/mul_1/x:output:0-final_layer/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
:  
$final_layer/kernel/Regularizer/add_1AddV2&final_layer/kernel/Regularizer/add:z:0(final_layer/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: k
IdentityIdentityfinal_layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
NoOpNoOp#^final_layer/BiasAdd/ReadVariableOp"^final_layer/MatMul/ReadVariableOp2^final_layer/kernel/Regularizer/Abs/ReadVariableOp5^final_layer/kernel/Regularizer/Square/ReadVariableOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2H
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*®
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ?
final_layer0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ìZ

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
M__call__
*N&call_and_return_all_conditional_losses
O_default_save_signature"
_tf_keras_sequential
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
É
	#decay
$learning_rate
%momentum
&itermomentumEmomentumFmomentumGmomentumHmomentumImomentumJmomentumKmomentumL"
	optimizer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
'
X0"
trackable_list_wrapper
Ê
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
M__call__
O_default_save_signature
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
,
Yserving_default"
signature_map
 :	2layer1/kernel
:2layer1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
 :	d2layer2/kernel
:d2layer2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
:dd2layer3/kernel
:d2layer3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
$:"d2final_layer/kernel
:2final_layer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
X0"
trackable_list_wrapper
­
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
 trainable_variables
!regularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
@0"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
X0"
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Atotal
	Bcount
C	variables
D	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
A0
B1"
trackable_list_wrapper
-
C	variables"
_generic_user_object
+:)	2SGD/layer1/kernel/momentum
%:#2SGD/layer1/bias/momentum
+:)	d2SGD/layer2/kernel/momentum
$:"d2SGD/layer2/bias/momentum
*:(dd2SGD/layer3/kernel/momentum
$:"d2SGD/layer3/bias/momentum
/:-d2SGD/final_layer/kernel/momentum
):'2SGD/final_layer/bias/momentum
ú2÷
+__inference_sequential_layer_call_fn_332582
+__inference_sequential_layer_call_fn_332865
+__inference_sequential_layer_call_fn_332886
+__inference_sequential_layer_call_fn_332724À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
F__inference_sequential_layer_call_and_return_conditional_losses_332932
F__inference_sequential_layer_call_and_return_conditional_losses_332978
F__inference_sequential_layer_call_and_return_conditional_losses_332763
F__inference_sequential_layer_call_and_return_conditional_losses_332802À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÌBÉ
!__inference__wrapped_model_332458input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_layer1_layer_call_fn_332987¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_layer1_layer_call_and_return_conditional_losses_332998¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_layer2_layer_call_fn_333007¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_layer2_layer_call_and_return_conditional_losses_333018¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_layer3_layer_call_fn_333027¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_layer3_layer_call_and_return_conditional_losses_333038¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_final_layer_layer_call_fn_333062¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_final_layer_layer_call_and_return_conditional_losses_333087¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³2°
__inference_loss_fn_0_333107
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
ËBÈ
$__inference_signature_wrapper_332844input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!__inference__wrapped_model_332458w0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "9ª6
4
final_layer%"
final_layerÿÿÿÿÿÿÿÿÿ§
G__inference_final_layer_layer_call_and_return_conditional_losses_333087\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_final_layer_layer_call_fn_333062O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿ£
B__inference_layer1_layer_call_and_return_conditional_losses_332998]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 {
'__inference_layer1_layer_call_fn_332987P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
B__inference_layer2_layer_call_and_return_conditional_losses_333018]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 {
'__inference_layer2_layer_call_fn_333007P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿd¢
B__inference_layer3_layer_call_and_return_conditional_losses_333038\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 z
'__inference_layer3_layer_call_fn_333027O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿd;
__inference_loss_fn_0_333107¢

¢ 
ª " µ
F__inference_sequential_layer_call_and_return_conditional_losses_332763k8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
F__inference_sequential_layer_call_and_return_conditional_losses_332802k8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
F__inference_sequential_layer_call_and_return_conditional_losses_332932j7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
F__inference_sequential_layer_call_and_return_conditional_losses_332978j7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_sequential_layer_call_fn_332582^8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_332724^8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_332865]7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_332886]7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ«
$__inference_signature_wrapper_332844;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"9ª6
4
final_layer%"
final_layerÿÿÿÿÿÿÿÿÿ