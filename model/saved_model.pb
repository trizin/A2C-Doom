??

??
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
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

;
Elu
features"T
activations"T"
Ttype:
2
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
delete_old_dirsbool(?
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
actor_critic/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameactor_critic/conv2d/kernel
?
.actor_critic/conv2d/kernel/Read/ReadVariableOpReadVariableOpactor_critic/conv2d/kernel*&
_output_shapes
: *
dtype0
?
actor_critic/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameactor_critic/conv2d/bias
?
,actor_critic/conv2d/bias/Read/ReadVariableOpReadVariableOpactor_critic/conv2d/bias*
_output_shapes
: *
dtype0
?
actor_critic/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*-
shared_nameactor_critic/conv2d_1/kernel
?
0actor_critic/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpactor_critic/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
?
actor_critic/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameactor_critic/conv2d_1/bias
?
.actor_critic/conv2d_1/bias/Read/ReadVariableOpReadVariableOpactor_critic/conv2d_1/bias*
_output_shapes
:@*
dtype0
?
actor_critic/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*-
shared_nameactor_critic/conv2d_2/kernel
?
0actor_critic/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpactor_critic/conv2d_2/kernel*'
_output_shapes
:@?*
dtype0
?
actor_critic/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameactor_critic/conv2d_2/bias
?
.actor_critic/conv2d_2/bias/Read/ReadVariableOpReadVariableOpactor_critic/conv2d_2/bias*
_output_shapes	
:?*
dtype0
?
actor_critic/batch_norm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameactor_critic/batch_norm/gamma
?
1actor_critic/batch_norm/gamma/Read/ReadVariableOpReadVariableOpactor_critic/batch_norm/gamma*
_output_shapes
: *
dtype0
?
actor_critic/batch_norm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameactor_critic/batch_norm/beta
?
0actor_critic/batch_norm/beta/Read/ReadVariableOpReadVariableOpactor_critic/batch_norm/beta*
_output_shapes
: *
dtype0
?
#actor_critic/batch_norm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#actor_critic/batch_norm/moving_mean
?
7actor_critic/batch_norm/moving_mean/Read/ReadVariableOpReadVariableOp#actor_critic/batch_norm/moving_mean*
_output_shapes
: *
dtype0
?
'actor_critic/batch_norm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'actor_critic/batch_norm/moving_variance
?
;actor_critic/batch_norm/moving_variance/Read/ReadVariableOpReadVariableOp'actor_critic/batch_norm/moving_variance*
_output_shapes
: *
dtype0
?
actor_critic/batch_norm/gamma_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!actor_critic/batch_norm/gamma_1
?
3actor_critic/batch_norm/gamma_1/Read/ReadVariableOpReadVariableOpactor_critic/batch_norm/gamma_1*
_output_shapes
:@*
dtype0
?
actor_critic/batch_norm/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name actor_critic/batch_norm/beta_1
?
2actor_critic/batch_norm/beta_1/Read/ReadVariableOpReadVariableOpactor_critic/batch_norm/beta_1*
_output_shapes
:@*
dtype0
?
%actor_critic/batch_norm/moving_mean_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%actor_critic/batch_norm/moving_mean_1
?
9actor_critic/batch_norm/moving_mean_1/Read/ReadVariableOpReadVariableOp%actor_critic/batch_norm/moving_mean_1*
_output_shapes
:@*
dtype0
?
)actor_critic/batch_norm/moving_variance_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)actor_critic/batch_norm/moving_variance_1
?
=actor_critic/batch_norm/moving_variance_1/Read/ReadVariableOpReadVariableOp)actor_critic/batch_norm/moving_variance_1*
_output_shapes
:@*
dtype0
?
actor_critic/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_nameactor_critic/dense/kernel
?
-actor_critic/dense/kernel/Read/ReadVariableOpReadVariableOpactor_critic/dense/kernel* 
_output_shapes
:
??*
dtype0
?
actor_critic/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameactor_critic/dense/bias
?
+actor_critic/dense/bias/Read/ReadVariableOpReadVariableOpactor_critic/dense/bias*
_output_shapes	
:?*
dtype0
?
actor_critic/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_nameactor_critic/dense_1/kernel
?
/actor_critic/dense_1/kernel/Read/ReadVariableOpReadVariableOpactor_critic/dense_1/kernel*
_output_shapes
:	?*
dtype0
?
actor_critic/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameactor_critic/dense_1/bias
?
-actor_critic/dense_1/bias/Read/ReadVariableOpReadVariableOpactor_critic/dense_1/bias*
_output_shapes
:*
dtype0
?
actor_critic/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_nameactor_critic/dense_2/kernel
?
/actor_critic/dense_2/kernel/Read/ReadVariableOpReadVariableOpactor_critic/dense_2/kernel*
_output_shapes
:	?*
dtype0
?
actor_critic/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameactor_critic/dense_2/bias
?
-actor_critic/dense_2/bias/Read/ReadVariableOpReadVariableOpactor_critic/dense_2/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?6
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?5
value?5B?5 B?5
?
	conv1

activation
	conv2
	conv3
normalization
normalization1
normalization2
flatten

	dense1
	
actor

critic

signatures
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
w
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
?

kernel
bias
# _self_saveable_object_factories
!regularization_losses
"	variables
#trainable_variables
$	keras_api
?

%kernel
&bias
#'_self_saveable_object_factories
(regularization_losses
)	variables
*trainable_variables
+	keras_api
?
,axis
	-gamma
.beta
/moving_mean
0moving_variance
#1_self_saveable_object_factories
2regularization_losses
3	variables
4trainable_variables
5	keras_api
?
6axis
	7gamma
8beta
9moving_mean
:moving_variance
#;_self_saveable_object_factories
<regularization_losses
=	variables
>trainable_variables
?	keras_api
w
#@_self_saveable_object_factories
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
w
#E_self_saveable_object_factories
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
?

Jkernel
Kbias
#L_self_saveable_object_factories
Mregularization_losses
N	variables
Otrainable_variables
P	keras_api
?

Qkernel
Rbias
#S_self_saveable_object_factories
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
?

Xkernel
Ybias
#Z_self_saveable_object_factories
[regularization_losses
\	variables
]trainable_variables
^	keras_api
 
 
 
?
0
1
2
3
%4
&5
-6
.7
/8
09
710
811
912
:13
J14
K15
Q16
R17
X18
Y19
v
0
1
2
3
%4
&5
-6
.7
78
89
J10
K11
Q12
R13
X14
Y15
?
_layer_metrics
`layer_regularization_losses
regularization_losses
anon_trainable_variables
	variables
trainable_variables

blayers
cmetrics
WU
VARIABLE_VALUEactor_critic/conv2d/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEactor_critic/conv2d/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
?
dlayer_metrics
elayer_regularization_losses
regularization_losses
fnon_trainable_variables
	variables
trainable_variables

glayers
hmetrics
 
 
 
 
?
ilayer_metrics
jlayer_regularization_losses
regularization_losses
knon_trainable_variables
	variables
trainable_variables

llayers
mmetrics
YW
VARIABLE_VALUEactor_critic/conv2d_1/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEactor_critic/conv2d_1/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
?
nlayer_metrics
olayer_regularization_losses
!regularization_losses
pnon_trainable_variables
"	variables
#trainable_variables

qlayers
rmetrics
YW
VARIABLE_VALUEactor_critic/conv2d_2/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEactor_critic/conv2d_2/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

%0
&1

%0
&1
?
slayer_metrics
tlayer_regularization_losses
(regularization_losses
unon_trainable_variables
)	variables
*trainable_variables

vlayers
wmetrics
 
a_
VARIABLE_VALUEactor_critic/batch_norm/gamma.normalization/gamma/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEactor_critic/batch_norm/beta-normalization/beta/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE#actor_critic/batch_norm/moving_mean4normalization/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE'actor_critic/batch_norm/moving_variance8normalization/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

-0
.1
/2
03

-0
.1
?
xlayer_metrics
ylayer_regularization_losses
2regularization_losses
znon_trainable_variables
3	variables
4trainable_variables

{layers
|metrics
 
db
VARIABLE_VALUEactor_critic/batch_norm/gamma_1/normalization1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEactor_critic/batch_norm/beta_1.normalization1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE%actor_critic/batch_norm/moving_mean_15normalization1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE)actor_critic/batch_norm/moving_variance_19normalization1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

70
81
92
:3

70
81
?
}layer_metrics
~layer_regularization_losses
<regularization_losses
non_trainable_variables
=	variables
>trainable_variables
?layers
?metrics
 
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
Aregularization_losses
?non_trainable_variables
B	variables
Ctrainable_variables
?layers
?metrics
 
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
Fregularization_losses
?non_trainable_variables
G	variables
Htrainable_variables
?layers
?metrics
WU
VARIABLE_VALUEactor_critic/dense/kernel(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEactor_critic/dense/bias&dense1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

J0
K1

J0
K1
?
?layer_metrics
 ?layer_regularization_losses
Mregularization_losses
?non_trainable_variables
N	variables
Otrainable_variables
?layers
?metrics
XV
VARIABLE_VALUEactor_critic/dense_1/kernel'actor/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEactor_critic/dense_1/bias%actor/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

Q0
R1

Q0
R1
?
?layer_metrics
 ?layer_regularization_losses
Tregularization_losses
?non_trainable_variables
U	variables
Vtrainable_variables
?layers
?metrics
YW
VARIABLE_VALUEactor_critic/dense_2/kernel(critic/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEactor_critic/dense_2/bias&critic/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

X0
Y1

X0
Y1
?
?layer_metrics
 ?layer_regularization_losses
[regularization_losses
?non_trainable_variables
\	variables
]trainable_variables
?layers
?metrics
 
 

/0
01
92
:3
N
0
1
2
3
4
5
6
7
	8

9
10
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

/0
01
 
 
 
 

90
:1
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
 
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????TT*
dtype0*$
shape:?????????TT
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1actor_critic/conv2d/kernelactor_critic/conv2d/biasactor_critic/batch_norm/gammaactor_critic/batch_norm/beta#actor_critic/batch_norm/moving_mean'actor_critic/batch_norm/moving_varianceactor_critic/conv2d_1/kernelactor_critic/conv2d_1/biasactor_critic/batch_norm/gamma_1actor_critic/batch_norm/beta_1%actor_critic/batch_norm/moving_mean_1)actor_critic/batch_norm/moving_variance_1actor_critic/conv2d_2/kernelactor_critic/conv2d_2/biasactor_critic/dense/kernelactor_critic/dense/biasactor_critic/dense_1/kernelactor_critic/dense_1/biasactor_critic/dense_2/kernelactor_critic/dense_2/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_3949386
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.actor_critic/conv2d/kernel/Read/ReadVariableOp,actor_critic/conv2d/bias/Read/ReadVariableOp0actor_critic/conv2d_1/kernel/Read/ReadVariableOp.actor_critic/conv2d_1/bias/Read/ReadVariableOp0actor_critic/conv2d_2/kernel/Read/ReadVariableOp.actor_critic/conv2d_2/bias/Read/ReadVariableOp1actor_critic/batch_norm/gamma/Read/ReadVariableOp0actor_critic/batch_norm/beta/Read/ReadVariableOp7actor_critic/batch_norm/moving_mean/Read/ReadVariableOp;actor_critic/batch_norm/moving_variance/Read/ReadVariableOp3actor_critic/batch_norm/gamma_1/Read/ReadVariableOp2actor_critic/batch_norm/beta_1/Read/ReadVariableOp9actor_critic/batch_norm/moving_mean_1/Read/ReadVariableOp=actor_critic/batch_norm/moving_variance_1/Read/ReadVariableOp-actor_critic/dense/kernel/Read/ReadVariableOp+actor_critic/dense/bias/Read/ReadVariableOp/actor_critic/dense_1/kernel/Read/ReadVariableOp-actor_critic/dense_1/bias/Read/ReadVariableOp/actor_critic/dense_2/kernel/Read/ReadVariableOp-actor_critic/dense_2/bias/Read/ReadVariableOpConst*!
Tin
2*
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
GPU 2J 8? *)
f$R"
 __inference__traced_save_3949806
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameactor_critic/conv2d/kernelactor_critic/conv2d/biasactor_critic/conv2d_1/kernelactor_critic/conv2d_1/biasactor_critic/conv2d_2/kernelactor_critic/conv2d_2/biasactor_critic/batch_norm/gammaactor_critic/batch_norm/beta#actor_critic/batch_norm/moving_mean'actor_critic/batch_norm/moving_varianceactor_critic/batch_norm/gamma_1actor_critic/batch_norm/beta_1%actor_critic/batch_norm/moving_mean_1)actor_critic/batch_norm/moving_variance_1actor_critic/dense/kernelactor_critic/dense/biasactor_critic/dense_1/kernelactor_critic/dense_1/biasactor_critic/dense_2/kernelactor_critic/dense_2/bias* 
Tin
2*
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
GPU 2J 8? *,
f'R%
#__inference__traced_restore_3949876??
?	
?
?__inference_conv2d_layer_call_and_return_conditional_losses_481

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????TT::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????TT
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_3949386
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_39493372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*~
_input_shapesm
k:?????????TT::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????TT
!
_user_specified_name	input_1
?
?
,__inference_batch_norm_layer_call_fn_3949709

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_batch_norm_layer_call_and_return_conditional_losses_39495522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
G__inference_batch_norm_layer_call_and_return_conditional_losses_3949448

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%??'7*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
G__inference_batch_norm_layer_call_and_return_conditional_losses_3949696

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
A__inference_conv2d_2_layer_call_and_return_conditional_losses_491

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
>__inference_dense_layer_call_and_return_conditional_losses_811

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_batch_norm_layer_call_fn_3949645

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_batch_norm_layer_call_and_return_conditional_losses_39494482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
X
<__inference_elu_layer_call_and_return_conditional_losses_502

inputs
identityS
EluEluinputs*
T0*/
_output_shapes
:?????????@2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_actor_critic_layer_call_fn_975

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_actor_critic_layer_call_and_return_conditional_losses_8942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*~
_input_shapesm
k:?????????TT::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????TT
 
_user_specified_nameinputs
?n
?
E__inference_actor_critic_layer_call_and_return_conditional_losses_790

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource&
"batch_norm_readvariableop_resource(
$batch_norm_readvariableop_1_resource7
3batch_norm_fusedbatchnormv3_readvariableop_resource9
5batch_norm_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$batch_norm_readvariableop_2_resource(
$batch_norm_readvariableop_3_resource9
5batch_norm_fusedbatchnormv3_1_readvariableop_resource;
7batch_norm_fusedbatchnormv3_1_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity

identity_1??batch_norm/AssignNewValue?batch_norm/AssignNewValue_1?batch_norm/AssignNewValue_2?batch_norm/AssignNewValue_3?*batch_norm/FusedBatchNormV3/ReadVariableOp?,batch_norm/FusedBatchNormV3/ReadVariableOp_1?,batch_norm/FusedBatchNormV3_1/ReadVariableOp?.batch_norm/FusedBatchNormV3_1/ReadVariableOp_1?batch_norm/ReadVariableOp?batch_norm/ReadVariableOp_1?batch_norm/ReadVariableOp_2?batch_norm/ReadVariableOp_3?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d/BiasAdd?
batch_norm/ReadVariableOpReadVariableOp"batch_norm_readvariableop_resource*
_output_shapes
: *
dtype02
batch_norm/ReadVariableOp?
batch_norm/ReadVariableOp_1ReadVariableOp$batch_norm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batch_norm/ReadVariableOp_1?
*batch_norm/FusedBatchNormV3/ReadVariableOpReadVariableOp3batch_norm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02,
*batch_norm/FusedBatchNormV3/ReadVariableOp?
,batch_norm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5batch_norm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02.
,batch_norm/FusedBatchNormV3/ReadVariableOp_1?
batch_norm/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0!batch_norm/ReadVariableOp:value:0#batch_norm/ReadVariableOp_1:value:02batch_norm/FusedBatchNormV3/ReadVariableOp:value:04batch_norm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%??'7*
exponential_avg_factor%
?#<2
batch_norm/FusedBatchNormV3?
batch_norm/AssignNewValueAssignVariableOp3batch_norm_fusedbatchnormv3_readvariableop_resource(batch_norm/FusedBatchNormV3:batch_mean:0+^batch_norm/FusedBatchNormV3/ReadVariableOp*F
_class<
:8loc:@batch_norm/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batch_norm/AssignNewValue?
batch_norm/AssignNewValue_1AssignVariableOp5batch_norm_fusedbatchnormv3_readvariableop_1_resource,batch_norm/FusedBatchNormV3:batch_variance:0-^batch_norm/FusedBatchNormV3/ReadVariableOp_1*H
_class>
<:loc:@batch_norm/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batch_norm/AssignNewValue_1t
elu/EluElubatch_norm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2	
elu/Elu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dbatch_norm/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd?
batch_norm/ReadVariableOp_2ReadVariableOp$batch_norm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batch_norm/ReadVariableOp_2?
batch_norm/ReadVariableOp_3ReadVariableOp$batch_norm_readvariableop_3_resource*
_output_shapes
:@*
dtype02
batch_norm/ReadVariableOp_3?
,batch_norm/FusedBatchNormV3_1/ReadVariableOpReadVariableOp5batch_norm_fusedbatchnormv3_1_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_norm/FusedBatchNormV3_1/ReadVariableOp?
.batch_norm/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOp7batch_norm_fusedbatchnormv3_1_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.batch_norm/FusedBatchNormV3_1/ReadVariableOp_1?
batch_norm/FusedBatchNormV3_1FusedBatchNormV3conv2d_1/BiasAdd:output:0#batch_norm/ReadVariableOp_2:value:0#batch_norm/ReadVariableOp_3:value:04batch_norm/FusedBatchNormV3_1/ReadVariableOp:value:06batch_norm/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%
?#<2
batch_norm/FusedBatchNormV3_1?
batch_norm/AssignNewValue_2AssignVariableOp5batch_norm_fusedbatchnormv3_1_readvariableop_resource*batch_norm/FusedBatchNormV3_1:batch_mean:0-^batch_norm/FusedBatchNormV3_1/ReadVariableOp*H
_class>
<:loc:@batch_norm/FusedBatchNormV3_1/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batch_norm/AssignNewValue_2?
batch_norm/AssignNewValue_3AssignVariableOp7batch_norm_fusedbatchnormv3_1_readvariableop_1_resource.batch_norm/FusedBatchNormV3_1:batch_variance:0/^batch_norm/FusedBatchNormV3_1/ReadVariableOp_1*J
_class@
><loc:@batch_norm/FusedBatchNormV3_1/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batch_norm/AssignNewValue_3z
	elu/Elu_1Elu!batch_norm/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:?????????@2
	elu/Elu_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D!batch_norm/FusedBatchNormV3_1:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdds
	elu/Elu_2Eluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
	elu/Elu_2o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapeconv2d_2/BiasAdd:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdd?
IdentityIdentitydense_1/BiasAdd:output:0^batch_norm/AssignNewValue^batch_norm/AssignNewValue_1^batch_norm/AssignNewValue_2^batch_norm/AssignNewValue_3+^batch_norm/FusedBatchNormV3/ReadVariableOp-^batch_norm/FusedBatchNormV3/ReadVariableOp_1-^batch_norm/FusedBatchNormV3_1/ReadVariableOp/^batch_norm/FusedBatchNormV3_1/ReadVariableOp_1^batch_norm/ReadVariableOp^batch_norm/ReadVariableOp_1^batch_norm/ReadVariableOp_2^batch_norm/ReadVariableOp_3^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_2/BiasAdd:output:0^batch_norm/AssignNewValue^batch_norm/AssignNewValue_1^batch_norm/AssignNewValue_2^batch_norm/AssignNewValue_3+^batch_norm/FusedBatchNormV3/ReadVariableOp-^batch_norm/FusedBatchNormV3/ReadVariableOp_1-^batch_norm/FusedBatchNormV3_1/ReadVariableOp/^batch_norm/FusedBatchNormV3_1/ReadVariableOp_1^batch_norm/ReadVariableOp^batch_norm/ReadVariableOp_1^batch_norm/ReadVariableOp_2^batch_norm/ReadVariableOp_3^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*~
_input_shapesm
k:?????????TT::::::::::::::::::::26
batch_norm/AssignNewValuebatch_norm/AssignNewValue2:
batch_norm/AssignNewValue_1batch_norm/AssignNewValue_12:
batch_norm/AssignNewValue_2batch_norm/AssignNewValue_22:
batch_norm/AssignNewValue_3batch_norm/AssignNewValue_32X
*batch_norm/FusedBatchNormV3/ReadVariableOp*batch_norm/FusedBatchNormV3/ReadVariableOp2\
,batch_norm/FusedBatchNormV3/ReadVariableOp_1,batch_norm/FusedBatchNormV3/ReadVariableOp_12\
,batch_norm/FusedBatchNormV3_1/ReadVariableOp,batch_norm/FusedBatchNormV3_1/ReadVariableOp2`
.batch_norm/FusedBatchNormV3_1/ReadVariableOp_1.batch_norm/FusedBatchNormV3_1/ReadVariableOp_126
batch_norm/ReadVariableOpbatch_norm/ReadVariableOp2:
batch_norm/ReadVariableOp_1batch_norm/ReadVariableOp_12:
batch_norm/ReadVariableOp_2batch_norm/ReadVariableOp_22:
batch_norm/ReadVariableOp_3batch_norm/ReadVariableOp_32>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????TT
 
_user_specified_nameinputs
?
?
G__inference_batch_norm_layer_call_and_return_conditional_losses_3949479

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
C__inference_batch_norm_layer_call_and_return_conditional_losses_857

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
G__inference_batch_norm_layer_call_and_return_conditional_losses_3949632

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
\
@__inference_flatten_layer_call_and_return_conditional_losses_497

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_batch_norm_layer_call_and_return_conditional_losses_3949552

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
,__inference_batch_norm_layer_call_fn_3949722

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_batch_norm_layer_call_and_return_conditional_losses_39495832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?n
?
E__inference_actor_critic_layer_call_and_return_conditional_losses_577
input_1)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource&
"batch_norm_readvariableop_resource(
$batch_norm_readvariableop_1_resource7
3batch_norm_fusedbatchnormv3_readvariableop_resource9
5batch_norm_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$batch_norm_readvariableop_2_resource(
$batch_norm_readvariableop_3_resource9
5batch_norm_fusedbatchnormv3_1_readvariableop_resource;
7batch_norm_fusedbatchnormv3_1_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity

identity_1??batch_norm/AssignNewValue?batch_norm/AssignNewValue_1?batch_norm/AssignNewValue_2?batch_norm/AssignNewValue_3?*batch_norm/FusedBatchNormV3/ReadVariableOp?,batch_norm/FusedBatchNormV3/ReadVariableOp_1?,batch_norm/FusedBatchNormV3_1/ReadVariableOp?.batch_norm/FusedBatchNormV3_1/ReadVariableOp_1?batch_norm/ReadVariableOp?batch_norm/ReadVariableOp_1?batch_norm/ReadVariableOp_2?batch_norm/ReadVariableOp_3?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinput_1$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d/BiasAdd?
batch_norm/ReadVariableOpReadVariableOp"batch_norm_readvariableop_resource*
_output_shapes
: *
dtype02
batch_norm/ReadVariableOp?
batch_norm/ReadVariableOp_1ReadVariableOp$batch_norm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batch_norm/ReadVariableOp_1?
*batch_norm/FusedBatchNormV3/ReadVariableOpReadVariableOp3batch_norm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02,
*batch_norm/FusedBatchNormV3/ReadVariableOp?
,batch_norm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5batch_norm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02.
,batch_norm/FusedBatchNormV3/ReadVariableOp_1?
batch_norm/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0!batch_norm/ReadVariableOp:value:0#batch_norm/ReadVariableOp_1:value:02batch_norm/FusedBatchNormV3/ReadVariableOp:value:04batch_norm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%??'7*
exponential_avg_factor%
?#<2
batch_norm/FusedBatchNormV3?
batch_norm/AssignNewValueAssignVariableOp3batch_norm_fusedbatchnormv3_readvariableop_resource(batch_norm/FusedBatchNormV3:batch_mean:0+^batch_norm/FusedBatchNormV3/ReadVariableOp*F
_class<
:8loc:@batch_norm/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batch_norm/AssignNewValue?
batch_norm/AssignNewValue_1AssignVariableOp5batch_norm_fusedbatchnormv3_readvariableop_1_resource,batch_norm/FusedBatchNormV3:batch_variance:0-^batch_norm/FusedBatchNormV3/ReadVariableOp_1*H
_class>
<:loc:@batch_norm/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batch_norm/AssignNewValue_1t
elu/EluElubatch_norm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2	
elu/Elu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dbatch_norm/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd?
batch_norm/ReadVariableOp_2ReadVariableOp$batch_norm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batch_norm/ReadVariableOp_2?
batch_norm/ReadVariableOp_3ReadVariableOp$batch_norm_readvariableop_3_resource*
_output_shapes
:@*
dtype02
batch_norm/ReadVariableOp_3?
,batch_norm/FusedBatchNormV3_1/ReadVariableOpReadVariableOp5batch_norm_fusedbatchnormv3_1_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_norm/FusedBatchNormV3_1/ReadVariableOp?
.batch_norm/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOp7batch_norm_fusedbatchnormv3_1_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.batch_norm/FusedBatchNormV3_1/ReadVariableOp_1?
batch_norm/FusedBatchNormV3_1FusedBatchNormV3conv2d_1/BiasAdd:output:0#batch_norm/ReadVariableOp_2:value:0#batch_norm/ReadVariableOp_3:value:04batch_norm/FusedBatchNormV3_1/ReadVariableOp:value:06batch_norm/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%
?#<2
batch_norm/FusedBatchNormV3_1?
batch_norm/AssignNewValue_2AssignVariableOp5batch_norm_fusedbatchnormv3_1_readvariableop_resource*batch_norm/FusedBatchNormV3_1:batch_mean:0-^batch_norm/FusedBatchNormV3_1/ReadVariableOp*H
_class>
<:loc:@batch_norm/FusedBatchNormV3_1/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batch_norm/AssignNewValue_2?
batch_norm/AssignNewValue_3AssignVariableOp7batch_norm_fusedbatchnormv3_1_readvariableop_1_resource.batch_norm/FusedBatchNormV3_1:batch_variance:0/^batch_norm/FusedBatchNormV3_1/ReadVariableOp_1*J
_class@
><loc:@batch_norm/FusedBatchNormV3_1/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batch_norm/AssignNewValue_3z
	elu/Elu_1Elu!batch_norm/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:?????????@2
	elu/Elu_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D!batch_norm/FusedBatchNormV3_1:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdds
	elu/Elu_2Eluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
	elu/Elu_2o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapeconv2d_2/BiasAdd:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdd?
IdentityIdentitydense_1/BiasAdd:output:0^batch_norm/AssignNewValue^batch_norm/AssignNewValue_1^batch_norm/AssignNewValue_2^batch_norm/AssignNewValue_3+^batch_norm/FusedBatchNormV3/ReadVariableOp-^batch_norm/FusedBatchNormV3/ReadVariableOp_1-^batch_norm/FusedBatchNormV3_1/ReadVariableOp/^batch_norm/FusedBatchNormV3_1/ReadVariableOp_1^batch_norm/ReadVariableOp^batch_norm/ReadVariableOp_1^batch_norm/ReadVariableOp_2^batch_norm/ReadVariableOp_3^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_2/BiasAdd:output:0^batch_norm/AssignNewValue^batch_norm/AssignNewValue_1^batch_norm/AssignNewValue_2^batch_norm/AssignNewValue_3+^batch_norm/FusedBatchNormV3/ReadVariableOp-^batch_norm/FusedBatchNormV3/ReadVariableOp_1-^batch_norm/FusedBatchNormV3_1/ReadVariableOp/^batch_norm/FusedBatchNormV3_1/ReadVariableOp_1^batch_norm/ReadVariableOp^batch_norm/ReadVariableOp_1^batch_norm/ReadVariableOp_2^batch_norm/ReadVariableOp_3^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*~
_input_shapesm
k:?????????TT::::::::::::::::::::26
batch_norm/AssignNewValuebatch_norm/AssignNewValue2:
batch_norm/AssignNewValue_1batch_norm/AssignNewValue_12:
batch_norm/AssignNewValue_2batch_norm/AssignNewValue_22:
batch_norm/AssignNewValue_3batch_norm/AssignNewValue_32X
*batch_norm/FusedBatchNormV3/ReadVariableOp*batch_norm/FusedBatchNormV3/ReadVariableOp2\
,batch_norm/FusedBatchNormV3/ReadVariableOp_1,batch_norm/FusedBatchNormV3/ReadVariableOp_12\
,batch_norm/FusedBatchNormV3_1/ReadVariableOp,batch_norm/FusedBatchNormV3_1/ReadVariableOp2`
.batch_norm/FusedBatchNormV3_1/ReadVariableOp_1.batch_norm/FusedBatchNormV3_1/ReadVariableOp_126
batch_norm/ReadVariableOpbatch_norm/ReadVariableOp2:
batch_norm/ReadVariableOp_1batch_norm/ReadVariableOp_12:
batch_norm/ReadVariableOp_2batch_norm/ReadVariableOp_22:
batch_norm/ReadVariableOp_3batch_norm/ReadVariableOp_32>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????TT
!
_user_specified_name	input_1
?
?
G__inference_batch_norm_layer_call_and_return_conditional_losses_3949614

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%??'7*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?;
?
E__inference_actor_critic_layer_call_and_return_conditional_losses_894

inputs
conv2d_15041054
conv2d_15041056
batch_norm_15041059
batch_norm_15041061
batch_norm_15041063
batch_norm_15041065
conv2d_1_15041069
conv2d_1_15041071
batch_norm_15041074
batch_norm_15041076
batch_norm_15041078
batch_norm_15041080
conv2d_2_15041084
conv2d_2_15041086
dense_15041091
dense_15041093
dense_1_15041096
dense_1_15041098
dense_2_15041101
dense_2_15041103
identity

identity_1??"batch_norm/StatefulPartitionedCall?$batch_norm/StatefulPartitionedCall_1?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_15041054conv2d_15041056*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_4812 
conv2d/StatefulPartitionedCall?
"batch_norm/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_norm_15041059batch_norm_15041061batch_norm_15041063batch_norm_15041065*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_batch_norm_layer_call_and_return_conditional_losses_8572$
"batch_norm/StatefulPartitionedCall?
elu/PartitionedCallPartitionedCall+batch_norm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_elu_layer_call_and_return_conditional_losses_3022
elu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall+batch_norm/StatefulPartitionedCall:output:0conv2d_1_15041069conv2d_1_15041071*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_4172"
 conv2d_1/StatefulPartitionedCall?
$batch_norm/StatefulPartitionedCall_1StatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_norm_15041074batch_norm_15041076batch_norm_15041078batch_norm_15041080*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_batch_norm_layer_call_and_return_conditional_losses_3892&
$batch_norm/StatefulPartitionedCall_1?
elu/PartitionedCall_1PartitionedCall-batch_norm/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_elu_layer_call_and_return_conditional_losses_5022
elu/PartitionedCall_1?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall-batch_norm/StatefulPartitionedCall_1:output:0conv2d_2_15041084conv2d_2_15041086*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_4912"
 conv2d_2/StatefulPartitionedCall?
elu/PartitionedCall_2PartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *E
f@R>
<__inference_elu_layer_call_and_return_conditional_losses_3072
elu/PartitionedCall_2?
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_4972
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_15041091dense_15041093*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_8112
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_15041096dense_1_15041098*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_8002!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_2_15041101dense_2_15041103*
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
GPU 2J 8? *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_8392!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0#^batch_norm/StatefulPartitionedCall%^batch_norm/StatefulPartitionedCall_1^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity(dense_2/StatefulPartitionedCall:output:0#^batch_norm/StatefulPartitionedCall%^batch_norm/StatefulPartitionedCall_1^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*~
_input_shapesm
k:?????????TT::::::::::::::::::::2H
"batch_norm/StatefulPartitionedCall"batch_norm/StatefulPartitionedCall2L
$batch_norm/StatefulPartitionedCall_1$batch_norm/StatefulPartitionedCall_12@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????TT
 
_user_specified_nameinputs
?	
?
A__inference_conv2d_1_layer_call_and_return_conditional_losses_417

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
*__inference_actor_critic_layer_call_fn_921
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_actor_critic_layer_call_and_return_conditional_losses_8942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*~
_input_shapesm
k:?????????TT::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????TT
!
_user_specified_name	input_1
?
?
G__inference_batch_norm_layer_call_and_return_conditional_losses_3949678

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
G__inference_batch_norm_layer_call_and_return_conditional_losses_3949583

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
@__inference_dense_2_layer_call_and_return_conditional_losses_839

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_restored_function_body_4809

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*:
_output_shapes(
&:?????????:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_actor_critic_layer_call_and_return_conditional_losses_2162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*~
_input_shapesm
k:?????????TT::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????TT
 
_user_specified_nameinputs
?
?
+__inference_actor_critic_layer_call_fn_1002

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_actor_critic_layer_call_and_return_conditional_losses_8942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*~
_input_shapesm
k:?????????TT::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????TT
 
_user_specified_nameinputs
?
?
"__inference__wrapped_model_3949337
input_1
actor_critic_3949293
actor_critic_3949295
actor_critic_3949297
actor_critic_3949299
actor_critic_3949301
actor_critic_3949303
actor_critic_3949305
actor_critic_3949307
actor_critic_3949309
actor_critic_3949311
actor_critic_3949313
actor_critic_3949315
actor_critic_3949317
actor_critic_3949319
actor_critic_3949321
actor_critic_3949323
actor_critic_3949325
actor_critic_3949327
actor_critic_3949329
actor_critic_3949331
identity

identity_1??$actor_critic/StatefulPartitionedCall?
$actor_critic/StatefulPartitionedCallStatefulPartitionedCallinput_1actor_critic_3949293actor_critic_3949295actor_critic_3949297actor_critic_3949299actor_critic_3949301actor_critic_3949303actor_critic_3949305actor_critic_3949307actor_critic_3949309actor_critic_3949311actor_critic_3949313actor_critic_3949315actor_critic_3949317actor_critic_3949319actor_critic_3949321actor_critic_3949323actor_critic_3949325actor_critic_3949327actor_critic_3949329actor_critic_3949331* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_48092&
$actor_critic/StatefulPartitionedCall?
IdentityIdentity-actor_critic/StatefulPartitionedCall:output:0%^actor_critic/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity-actor_critic/StatefulPartitionedCall:output:1%^actor_critic/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*~
_input_shapesm
k:?????????TT::::::::::::::::::::2L
$actor_critic/StatefulPartitionedCall$actor_critic/StatefulPartitionedCall:X T
/
_output_shapes
:?????????TT
!
_user_specified_name	input_1
?
?
,__inference_batch_norm_layer_call_fn_3949658

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_batch_norm_layer_call_and_return_conditional_losses_39494792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
@__inference_dense_1_layer_call_and_return_conditional_losses_800

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
X
<__inference_elu_layer_call_and_return_conditional_losses_302

inputs
identityS
EluEluinputs*
T0*/
_output_shapes
:????????? 2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?W
?
#__inference__traced_restore_3949876
file_prefix/
+assignvariableop_actor_critic_conv2d_kernel/
+assignvariableop_1_actor_critic_conv2d_bias3
/assignvariableop_2_actor_critic_conv2d_1_kernel1
-assignvariableop_3_actor_critic_conv2d_1_bias3
/assignvariableop_4_actor_critic_conv2d_2_kernel1
-assignvariableop_5_actor_critic_conv2d_2_bias4
0assignvariableop_6_actor_critic_batch_norm_gamma3
/assignvariableop_7_actor_critic_batch_norm_beta:
6assignvariableop_8_actor_critic_batch_norm_moving_mean>
:assignvariableop_9_actor_critic_batch_norm_moving_variance7
3assignvariableop_10_actor_critic_batch_norm_gamma_16
2assignvariableop_11_actor_critic_batch_norm_beta_1=
9assignvariableop_12_actor_critic_batch_norm_moving_mean_1A
=assignvariableop_13_actor_critic_batch_norm_moving_variance_11
-assignvariableop_14_actor_critic_dense_kernel/
+assignvariableop_15_actor_critic_dense_bias3
/assignvariableop_16_actor_critic_dense_1_kernel1
-assignvariableop_17_actor_critic_dense_1_bias3
/assignvariableop_18_actor_critic_dense_2_kernel1
-assignvariableop_19_actor_critic_dense_2_bias
identity_21??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB.normalization/gamma/.ATTRIBUTES/VARIABLE_VALUEB-normalization/beta/.ATTRIBUTES/VARIABLE_VALUEB4normalization/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB8normalization/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB/normalization1/gamma/.ATTRIBUTES/VARIABLE_VALUEB.normalization1/beta/.ATTRIBUTES/VARIABLE_VALUEB5normalization1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB9normalization1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB'actor/kernel/.ATTRIBUTES/VARIABLE_VALUEB%actor/bias/.ATTRIBUTES/VARIABLE_VALUEB(critic/kernel/.ATTRIBUTES/VARIABLE_VALUEB&critic/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp+assignvariableop_actor_critic_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp+assignvariableop_1_actor_critic_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_actor_critic_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp-assignvariableop_3_actor_critic_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp/assignvariableop_4_actor_critic_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp-assignvariableop_5_actor_critic_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp0assignvariableop_6_actor_critic_batch_norm_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp/assignvariableop_7_actor_critic_batch_norm_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp6assignvariableop_8_actor_critic_batch_norm_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp:assignvariableop_9_actor_critic_batch_norm_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp3assignvariableop_10_actor_critic_batch_norm_gamma_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp2assignvariableop_11_actor_critic_batch_norm_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp9assignvariableop_12_actor_critic_batch_norm_moving_mean_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp=assignvariableop_13_actor_critic_batch_norm_moving_variance_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp-assignvariableop_14_actor_critic_dense_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp+assignvariableop_15_actor_critic_dense_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp/assignvariableop_16_actor_critic_dense_1_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp-assignvariableop_17_actor_critic_dense_1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp/assignvariableop_18_actor_critic_dense_2_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp-assignvariableop_19_actor_critic_dense_2_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_199
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20?
Identity_21IdentityIdentity_20:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_21"#
identity_21Identity_21:output:0*e
_input_shapesT
R: ::::::::::::::::::::2$
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
AssignVariableOp_2AssignVariableOp_22(
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
?_
?
E__inference_actor_critic_layer_call_and_return_conditional_losses_216

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource&
"batch_norm_readvariableop_resource(
$batch_norm_readvariableop_1_resource7
3batch_norm_fusedbatchnormv3_readvariableop_resource9
5batch_norm_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$batch_norm_readvariableop_2_resource(
$batch_norm_readvariableop_3_resource9
5batch_norm_fusedbatchnormv3_1_readvariableop_resource;
7batch_norm_fusedbatchnormv3_1_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity

identity_1??*batch_norm/FusedBatchNormV3/ReadVariableOp?,batch_norm/FusedBatchNormV3/ReadVariableOp_1?,batch_norm/FusedBatchNormV3_1/ReadVariableOp?.batch_norm/FusedBatchNormV3_1/ReadVariableOp_1?batch_norm/ReadVariableOp?batch_norm/ReadVariableOp_1?batch_norm/ReadVariableOp_2?batch_norm/ReadVariableOp_3?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d/BiasAdd?
batch_norm/ReadVariableOpReadVariableOp"batch_norm_readvariableop_resource*
_output_shapes
: *
dtype02
batch_norm/ReadVariableOp?
batch_norm/ReadVariableOp_1ReadVariableOp$batch_norm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batch_norm/ReadVariableOp_1?
*batch_norm/FusedBatchNormV3/ReadVariableOpReadVariableOp3batch_norm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02,
*batch_norm/FusedBatchNormV3/ReadVariableOp?
,batch_norm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5batch_norm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02.
,batch_norm/FusedBatchNormV3/ReadVariableOp_1?
batch_norm/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0!batch_norm/ReadVariableOp:value:0#batch_norm/ReadVariableOp_1:value:02batch_norm/FusedBatchNormV3/ReadVariableOp:value:04batch_norm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%??'7*
is_training( 2
batch_norm/FusedBatchNormV3t
elu/EluElubatch_norm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2	
elu/Elu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dbatch_norm/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd?
batch_norm/ReadVariableOp_2ReadVariableOp$batch_norm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batch_norm/ReadVariableOp_2?
batch_norm/ReadVariableOp_3ReadVariableOp$batch_norm_readvariableop_3_resource*
_output_shapes
:@*
dtype02
batch_norm/ReadVariableOp_3?
,batch_norm/FusedBatchNormV3_1/ReadVariableOpReadVariableOp5batch_norm_fusedbatchnormv3_1_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_norm/FusedBatchNormV3_1/ReadVariableOp?
.batch_norm/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOp7batch_norm_fusedbatchnormv3_1_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.batch_norm/FusedBatchNormV3_1/ReadVariableOp_1?
batch_norm/FusedBatchNormV3_1FusedBatchNormV3conv2d_1/BiasAdd:output:0#batch_norm/ReadVariableOp_2:value:0#batch_norm/ReadVariableOp_3:value:04batch_norm/FusedBatchNormV3_1/ReadVariableOp:value:06batch_norm/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
batch_norm/FusedBatchNormV3_1z
	elu/Elu_1Elu!batch_norm/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:?????????@2
	elu/Elu_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D!batch_norm/FusedBatchNormV3_1:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdds
	elu/Elu_2Eluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
	elu/Elu_2o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapeconv2d_2/BiasAdd:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdd?
IdentityIdentitydense_1/BiasAdd:output:0+^batch_norm/FusedBatchNormV3/ReadVariableOp-^batch_norm/FusedBatchNormV3/ReadVariableOp_1-^batch_norm/FusedBatchNormV3_1/ReadVariableOp/^batch_norm/FusedBatchNormV3_1/ReadVariableOp_1^batch_norm/ReadVariableOp^batch_norm/ReadVariableOp_1^batch_norm/ReadVariableOp_2^batch_norm/ReadVariableOp_3^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_2/BiasAdd:output:0+^batch_norm/FusedBatchNormV3/ReadVariableOp-^batch_norm/FusedBatchNormV3/ReadVariableOp_1-^batch_norm/FusedBatchNormV3_1/ReadVariableOp/^batch_norm/FusedBatchNormV3_1/ReadVariableOp_1^batch_norm/ReadVariableOp^batch_norm/ReadVariableOp_1^batch_norm/ReadVariableOp_2^batch_norm/ReadVariableOp_3^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*~
_input_shapesm
k:?????????TT::::::::::::::::::::2X
*batch_norm/FusedBatchNormV3/ReadVariableOp*batch_norm/FusedBatchNormV3/ReadVariableOp2\
,batch_norm/FusedBatchNormV3/ReadVariableOp_1,batch_norm/FusedBatchNormV3/ReadVariableOp_12\
,batch_norm/FusedBatchNormV3_1/ReadVariableOp,batch_norm/FusedBatchNormV3_1/ReadVariableOp2`
.batch_norm/FusedBatchNormV3_1/ReadVariableOp_1.batch_norm/FusedBatchNormV3_1/ReadVariableOp_126
batch_norm/ReadVariableOpbatch_norm/ReadVariableOp2:
batch_norm/ReadVariableOp_1batch_norm/ReadVariableOp_12:
batch_norm/ReadVariableOp_2batch_norm/ReadVariableOp_22:
batch_norm/ReadVariableOp_3batch_norm/ReadVariableOp_32>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????TT
 
_user_specified_nameinputs
?
?
*__inference_actor_critic_layer_call_fn_948
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_actor_critic_layer_call_and_return_conditional_losses_8942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*~
_input_shapesm
k:?????????TT::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????TT
!
_user_specified_name	input_1
?
?
C__inference_batch_norm_layer_call_and_return_conditional_losses_389

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?4
?

 __inference__traced_save_3949806
file_prefix9
5savev2_actor_critic_conv2d_kernel_read_readvariableop7
3savev2_actor_critic_conv2d_bias_read_readvariableop;
7savev2_actor_critic_conv2d_1_kernel_read_readvariableop9
5savev2_actor_critic_conv2d_1_bias_read_readvariableop;
7savev2_actor_critic_conv2d_2_kernel_read_readvariableop9
5savev2_actor_critic_conv2d_2_bias_read_readvariableop<
8savev2_actor_critic_batch_norm_gamma_read_readvariableop;
7savev2_actor_critic_batch_norm_beta_read_readvariableopB
>savev2_actor_critic_batch_norm_moving_mean_read_readvariableopF
Bsavev2_actor_critic_batch_norm_moving_variance_read_readvariableop>
:savev2_actor_critic_batch_norm_gamma_1_read_readvariableop=
9savev2_actor_critic_batch_norm_beta_1_read_readvariableopD
@savev2_actor_critic_batch_norm_moving_mean_1_read_readvariableopH
Dsavev2_actor_critic_batch_norm_moving_variance_1_read_readvariableop8
4savev2_actor_critic_dense_kernel_read_readvariableop6
2savev2_actor_critic_dense_bias_read_readvariableop:
6savev2_actor_critic_dense_1_kernel_read_readvariableop8
4savev2_actor_critic_dense_1_bias_read_readvariableop:
6savev2_actor_critic_dense_2_kernel_read_readvariableop8
4savev2_actor_critic_dense_2_bias_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB.normalization/gamma/.ATTRIBUTES/VARIABLE_VALUEB-normalization/beta/.ATTRIBUTES/VARIABLE_VALUEB4normalization/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB8normalization/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB/normalization1/gamma/.ATTRIBUTES/VARIABLE_VALUEB.normalization1/beta/.ATTRIBUTES/VARIABLE_VALUEB5normalization1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB9normalization1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB'actor/kernel/.ATTRIBUTES/VARIABLE_VALUEB%actor/bias/.ATTRIBUTES/VARIABLE_VALUEB(critic/kernel/.ATTRIBUTES/VARIABLE_VALUEB&critic/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_actor_critic_conv2d_kernel_read_readvariableop3savev2_actor_critic_conv2d_bias_read_readvariableop7savev2_actor_critic_conv2d_1_kernel_read_readvariableop5savev2_actor_critic_conv2d_1_bias_read_readvariableop7savev2_actor_critic_conv2d_2_kernel_read_readvariableop5savev2_actor_critic_conv2d_2_bias_read_readvariableop8savev2_actor_critic_batch_norm_gamma_read_readvariableop7savev2_actor_critic_batch_norm_beta_read_readvariableop>savev2_actor_critic_batch_norm_moving_mean_read_readvariableopBsavev2_actor_critic_batch_norm_moving_variance_read_readvariableop:savev2_actor_critic_batch_norm_gamma_1_read_readvariableop9savev2_actor_critic_batch_norm_beta_1_read_readvariableop@savev2_actor_critic_batch_norm_moving_mean_1_read_readvariableopDsavev2_actor_critic_batch_norm_moving_variance_1_read_readvariableop4savev2_actor_critic_dense_kernel_read_readvariableop2savev2_actor_critic_dense_bias_read_readvariableop6savev2_actor_critic_dense_1_kernel_read_readvariableop4savev2_actor_critic_dense_1_bias_read_readvariableop6savev2_actor_critic_dense_2_kernel_read_readvariableop4savev2_actor_critic_dense_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
22
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
?: : : : @:@:@?:?: : : : :@:@:@:@:
??:?:	?::	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?: 

_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
?_
?
E__inference_actor_critic_layer_call_and_return_conditional_losses_652
input_1)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource&
"batch_norm_readvariableop_resource(
$batch_norm_readvariableop_1_resource7
3batch_norm_fusedbatchnormv3_readvariableop_resource9
5batch_norm_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$batch_norm_readvariableop_2_resource(
$batch_norm_readvariableop_3_resource9
5batch_norm_fusedbatchnormv3_1_readvariableop_resource;
7batch_norm_fusedbatchnormv3_1_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity

identity_1??*batch_norm/FusedBatchNormV3/ReadVariableOp?,batch_norm/FusedBatchNormV3/ReadVariableOp_1?,batch_norm/FusedBatchNormV3_1/ReadVariableOp?.batch_norm/FusedBatchNormV3_1/ReadVariableOp_1?batch_norm/ReadVariableOp?batch_norm/ReadVariableOp_1?batch_norm/ReadVariableOp_2?batch_norm/ReadVariableOp_3?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinput_1$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d/BiasAdd?
batch_norm/ReadVariableOpReadVariableOp"batch_norm_readvariableop_resource*
_output_shapes
: *
dtype02
batch_norm/ReadVariableOp?
batch_norm/ReadVariableOp_1ReadVariableOp$batch_norm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batch_norm/ReadVariableOp_1?
*batch_norm/FusedBatchNormV3/ReadVariableOpReadVariableOp3batch_norm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02,
*batch_norm/FusedBatchNormV3/ReadVariableOp?
,batch_norm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5batch_norm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02.
,batch_norm/FusedBatchNormV3/ReadVariableOp_1?
batch_norm/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0!batch_norm/ReadVariableOp:value:0#batch_norm/ReadVariableOp_1:value:02batch_norm/FusedBatchNormV3/ReadVariableOp:value:04batch_norm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%??'7*
is_training( 2
batch_norm/FusedBatchNormV3t
elu/EluElubatch_norm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2	
elu/Elu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dbatch_norm/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd?
batch_norm/ReadVariableOp_2ReadVariableOp$batch_norm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batch_norm/ReadVariableOp_2?
batch_norm/ReadVariableOp_3ReadVariableOp$batch_norm_readvariableop_3_resource*
_output_shapes
:@*
dtype02
batch_norm/ReadVariableOp_3?
,batch_norm/FusedBatchNormV3_1/ReadVariableOpReadVariableOp5batch_norm_fusedbatchnormv3_1_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_norm/FusedBatchNormV3_1/ReadVariableOp?
.batch_norm/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOp7batch_norm_fusedbatchnormv3_1_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.batch_norm/FusedBatchNormV3_1/ReadVariableOp_1?
batch_norm/FusedBatchNormV3_1FusedBatchNormV3conv2d_1/BiasAdd:output:0#batch_norm/ReadVariableOp_2:value:0#batch_norm/ReadVariableOp_3:value:04batch_norm/FusedBatchNormV3_1/ReadVariableOp:value:06batch_norm/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
batch_norm/FusedBatchNormV3_1z
	elu/Elu_1Elu!batch_norm/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:?????????@2
	elu/Elu_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D!batch_norm/FusedBatchNormV3_1:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdds
	elu/Elu_2Eluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
	elu/Elu_2o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapeconv2d_2/BiasAdd:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdd?
IdentityIdentitydense_1/BiasAdd:output:0+^batch_norm/FusedBatchNormV3/ReadVariableOp-^batch_norm/FusedBatchNormV3/ReadVariableOp_1-^batch_norm/FusedBatchNormV3_1/ReadVariableOp/^batch_norm/FusedBatchNormV3_1/ReadVariableOp_1^batch_norm/ReadVariableOp^batch_norm/ReadVariableOp_1^batch_norm/ReadVariableOp_2^batch_norm/ReadVariableOp_3^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_2/BiasAdd:output:0+^batch_norm/FusedBatchNormV3/ReadVariableOp-^batch_norm/FusedBatchNormV3/ReadVariableOp_1-^batch_norm/FusedBatchNormV3_1/ReadVariableOp/^batch_norm/FusedBatchNormV3_1/ReadVariableOp_1^batch_norm/ReadVariableOp^batch_norm/ReadVariableOp_1^batch_norm/ReadVariableOp_2^batch_norm/ReadVariableOp_3^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*~
_input_shapesm
k:?????????TT::::::::::::::::::::2X
*batch_norm/FusedBatchNormV3/ReadVariableOp*batch_norm/FusedBatchNormV3/ReadVariableOp2\
,batch_norm/FusedBatchNormV3/ReadVariableOp_1,batch_norm/FusedBatchNormV3/ReadVariableOp_12\
,batch_norm/FusedBatchNormV3_1/ReadVariableOp,batch_norm/FusedBatchNormV3_1/ReadVariableOp2`
.batch_norm/FusedBatchNormV3_1/ReadVariableOp_1.batch_norm/FusedBatchNormV3_1/ReadVariableOp_126
batch_norm/ReadVariableOpbatch_norm/ReadVariableOp2:
batch_norm/ReadVariableOp_1batch_norm/ReadVariableOp_12:
batch_norm/ReadVariableOp_2batch_norm/ReadVariableOp_22:
batch_norm/ReadVariableOp_3batch_norm/ReadVariableOp_32>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????TT
!
_user_specified_name	input_1
?
X
<__inference_elu_layer_call_and_return_conditional_losses_307

inputs
identityT
EluEluinputs*
T0*0
_output_shapes
:??????????2
Elun
IdentityIdentityElu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????TT<
output_10
StatefulPartitionedCall:0?????????<
output_20
StatefulPartitionedCall:1?????????tensorflow/serving/predict:??
?
	conv1

activation
	conv2
	conv3
normalization
normalization1
normalization2
flatten

	dense1
	
actor

critic

signatures
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_model?{"class_name": "ActorCritic", "name": "actor_critic", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ActorCritic"}}
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 84, 84, 4]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 84, 84, 4]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 84, 84, 4]}}
?
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ELU", "name": "elu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "elu", "trainable": true, "dtype": "float32", "alpha": 1.0}}
?


kernel
bias
# _self_saveable_object_factories
!regularization_losses
"	variables
#trainable_variables
$	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 21, 21, 32]}}
?


%kernel
&bias
#'_self_saveable_object_factories
(regularization_losses
)	variables
*trainable_variables
+	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 11, 11, 64]}}
?	
,axis
	-gamma
.beta
/moving_mean
0moving_variance
#1_self_saveable_object_factories
2regularization_losses
3	variables
4trainable_variables
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_norm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_norm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 21, 21, 32]}}
?	
6axis
	7gamma
8beta
9moving_mean
:moving_variance
#;_self_saveable_object_factories
<regularization_losses
=	variables
>trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_norm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_norm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 11, 11, 64]}}
?
#@_self_saveable_object_factories
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_norm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_norm", "trainable": true, "dtype": "float32", "axis": -1, "momentum": 0.99, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}
?
#E_self_saveable_object_factories
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

Jkernel
Kbias
#L_self_saveable_object_factories
Mregularization_losses
N	variables
Otrainable_variables
P	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 3200]}}
?

Qkernel
Rbias
#S_self_saveable_object_factories
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
?

Xkernel
Ybias
#Z_self_saveable_object_factories
[regularization_losses
\	variables
]trainable_variables
^	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
%4
&5
-6
.7
/8
09
710
811
912
:13
J14
K15
Q16
R17
X18
Y19"
trackable_list_wrapper
?
0
1
2
3
%4
&5
-6
.7
78
89
J10
K11
Q12
R13
X14
Y15"
trackable_list_wrapper
?
_layer_metrics
`layer_regularization_losses
regularization_losses
anon_trainable_variables
	variables
trainable_variables

blayers
cmetrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:2 2actor_critic/conv2d/kernel
&:$ 2actor_critic/conv2d/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
dlayer_metrics
elayer_regularization_losses
regularization_losses
fnon_trainable_variables
	variables
trainable_variables

glayers
hmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ilayer_metrics
jlayer_regularization_losses
regularization_losses
knon_trainable_variables
	variables
trainable_variables

llayers
mmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
6:4 @2actor_critic/conv2d_1/kernel
(:&@2actor_critic/conv2d_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
nlayer_metrics
olayer_regularization_losses
!regularization_losses
pnon_trainable_variables
"	variables
#trainable_variables

qlayers
rmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
7:5@?2actor_critic/conv2d_2/kernel
):'?2actor_critic/conv2d_2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
slayer_metrics
tlayer_regularization_losses
(regularization_losses
unon_trainable_variables
)	variables
*trainable_variables

vlayers
wmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:) 2actor_critic/batch_norm/gamma
*:( 2actor_critic/batch_norm/beta
3:1  (2#actor_critic/batch_norm/moving_mean
7:5  (2'actor_critic/batch_norm/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
-0
.1
/2
03"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
?
xlayer_metrics
ylayer_regularization_losses
2regularization_losses
znon_trainable_variables
3	variables
4trainable_variables

{layers
|metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)@2actor_critic/batch_norm/gamma
*:(@2actor_critic/batch_norm/beta
3:1@ (2#actor_critic/batch_norm/moving_mean
7:5@ (2'actor_critic/batch_norm/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
70
81
92
:3"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
?
}layer_metrics
~layer_regularization_losses
<regularization_losses
non_trainable_variables
=	variables
>trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
Aregularization_losses
?non_trainable_variables
B	variables
Ctrainable_variables
?layers
?metrics"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
Fregularization_losses
?non_trainable_variables
G	variables
Htrainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+
??2actor_critic/dense/kernel
&:$?2actor_critic/dense/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
Mregularization_losses
?non_trainable_variables
N	variables
Otrainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,	?2actor_critic/dense_1/kernel
':%2actor_critic/dense_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
Tregularization_losses
?non_trainable_variables
U	variables
Vtrainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,	?2actor_critic/dense_2/kernel
':%2actor_critic/dense_2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
[regularization_losses
?non_trainable_variables
\	variables
]trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
/0
01
92
:3"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
90
:1"
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
?2?
"__inference__wrapped_model_3949337?
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
annotations? *.?+
)?&
input_1?????????TT
?2?
E__inference_actor_critic_layer_call_and_return_conditional_losses_577
E__inference_actor_critic_layer_call_and_return_conditional_losses_216
E__inference_actor_critic_layer_call_and_return_conditional_losses_652
E__inference_actor_critic_layer_call_and_return_conditional_losses_790?
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_actor_critic_layer_call_fn_921
*__inference_actor_critic_layer_call_fn_948
+__inference_actor_critic_layer_call_fn_1002
*__inference_actor_critic_layer_call_fn_975?
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2?
G__inference_batch_norm_layer_call_and_return_conditional_losses_3949614
G__inference_batch_norm_layer_call_and_return_conditional_losses_3949632?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_batch_norm_layer_call_fn_3949645
,__inference_batch_norm_layer_call_fn_3949658?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_batch_norm_layer_call_and_return_conditional_losses_3949678
G__inference_batch_norm_layer_call_and_return_conditional_losses_3949696?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_batch_norm_layer_call_fn_3949722
,__inference_batch_norm_layer_call_fn_3949709?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?B?
%__inference_signature_wrapper_3949386input_1"?
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
"__inference__wrapped_model_3949337?-./0789:%&JKQRXY8?5
.?+
)?&
input_1?????????TT
? "c?`
.
output_1"?
output_1?????????
.
output_2"?
output_2??????????
E__inference_actor_critic_layer_call_and_return_conditional_losses_216?-./0789:%&JKQRXY;?8
1?.
(?%
inputs?????????TT
p 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
E__inference_actor_critic_layer_call_and_return_conditional_losses_577?-./0789:%&JKQRXY<?9
2?/
)?&
input_1?????????TT
p
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
E__inference_actor_critic_layer_call_and_return_conditional_losses_652?-./0789:%&JKQRXY<?9
2?/
)?&
input_1?????????TT
p 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
E__inference_actor_critic_layer_call_and_return_conditional_losses_790?-./0789:%&JKQRXY;?8
1?.
(?%
inputs?????????TT
p
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
+__inference_actor_critic_layer_call_fn_1002?-./0789:%&JKQRXY;?8
1?.
(?%
inputs?????????TT
p
? "=?:
?
0?????????
?
1??????????
*__inference_actor_critic_layer_call_fn_921?-./0789:%&JKQRXY<?9
2?/
)?&
input_1?????????TT
p
? "=?:
?
0?????????
?
1??????????
*__inference_actor_critic_layer_call_fn_948?-./0789:%&JKQRXY<?9
2?/
)?&
input_1?????????TT
p 
? "=?:
?
0?????????
?
1??????????
*__inference_actor_critic_layer_call_fn_975?-./0789:%&JKQRXY;?8
1?.
(?%
inputs?????????TT
p 
? "=?:
?
0?????????
?
1??????????
G__inference_batch_norm_layer_call_and_return_conditional_losses_3949614?-./0M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
G__inference_batch_norm_layer_call_and_return_conditional_losses_3949632?-./0M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
G__inference_batch_norm_layer_call_and_return_conditional_losses_3949678?789:M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
G__inference_batch_norm_layer_call_and_return_conditional_losses_3949696?789:M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
,__inference_batch_norm_layer_call_fn_3949645?-./0M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
,__inference_batch_norm_layer_call_fn_3949658?-./0M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
,__inference_batch_norm_layer_call_fn_3949709?789:M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
,__inference_batch_norm_layer_call_fn_3949722?789:M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
%__inference_signature_wrapper_3949386?-./0789:%&JKQRXYC?@
? 
9?6
4
input_1)?&
input_1?????????TT"c?`
.
output_1"?
output_1?????????
.
output_2"?
output_2?????????