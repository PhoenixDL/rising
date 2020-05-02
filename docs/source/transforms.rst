.. role:: hidden
    :class: hidden-section

Data Transformations - rising.transforms
========================================

.. automodule:: rising.transforms

.. currentmodule:: rising.transforms

Module Interface
----------------

Transformation Base Classes
***************************

:hidden:`AbstractTransform`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: rising.transforms.abstract

.. autoclass:: AbstractTransform
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`BaseTransform`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BaseTransform
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`PerSampleTransform`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PerSampleTransform
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`PerChannelTransform`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PerChannelTransform
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`RandomDimsTransform`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomDimsTransform
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`RandomProcess`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomProcess
    :members:
    :undoc-members:
    :show-inheritance:

Compose
*******

.. currentmodule:: rising.transforms.compose

:hidden:`Compose`
~~~~~~~~~~~~~~~~~

.. autoclass:: Compose
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`DropoutCompose`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DropoutCompose
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`dict_call`
~~~~~~~~~~~~~~~~~~~

.. autofunction:: dict_call

Affine Transforms
****************

.. py:currentmodule:: rising.transforms.affine

:hidden:`Affine`
~~~~~~~~~~~~~~~

.. autoclass:: Affine
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`StackedAffine`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: StackedAffine
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`BaseAffine`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: BaseAffine
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`Rotate`
~~~~~~~~~~~~~~~~

.. autoclass:: Rotate
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`Translate`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: Translate
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`Scale`
~~~~~~~~~~~~~~~

.. autoclass:: Scale
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`Resize`
~~~~~~~~~~~~~~~~

.. autoclass:: Resize
    :members:
    :undoc-members:
    :show-inheritance:

Channel Transforms
******************

.. currentmodule:: rising.transforms.channel

:hidden:`OneHot`
~~~~~~~~~~~~~~~~

.. autoclass:: OneHot
    :members:
    :undoc-members:
    :show_inheritance:

Cropping Transforms
*******************

.. currentmodule:: rising.transforms.crop

:hidden:`CenterCrop`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CenterCrop
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`RandomCrop`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomCrop
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`CenterCropRandomSize`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CenterCropRandomSize
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`RandomCropRandomSize`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomCropRandomSize
    :members:
    :undoc-members:
    :show_inheritance:

Format Transforms
*****************

.. currentmodule:: rising.transforms.format

:hidden:`MapToSeq`
~~~~~~~~~~~~~~~~~~

.. autoclass:: MapToSeq
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`SeqToMap`
~~~~~~~~~~~~~~~~~~

.. autoclass:: SeqToMap
    :members:
    :undoc-members:
    :show_inheritance:

Intensity Transforms
********************

.. currentmodule:: rising.transforms.format

:hidden:`Clamp`
~~~~~~~~~~~~~~~

.. autoclass:: Clamp
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`NormRange`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: NormRange
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`NormMinMax`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NormMinMax
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`NormZeroMeanUnitStd`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NormZeroMeanUnitStd
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`NormMeanStd`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NormMeanStd
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`Noise`
~~~~~~~~~~~~~~~

.. autoclass:: Noise
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`GaussianNoise`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GaussianNoise
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`ExponentialNoise`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ExponentialNoise
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`GammaCorrection`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GammaCorrection
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`RandomValuePerChannel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomValuePerChannel
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`RandomAddValue`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomAddValue
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`RandomScaleValue`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomScaleValue
    :members:
    :undoc-members:
    :show_inheritance:

Kernel Transforms
*****************

.. currentmodule:: rising.transforms.kernel

:hidden:`KernelTransform`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: KernelTransform
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`GaussianSmoothing`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GaussianSmoothing
    :members:
    :undoc-members:
    :show_inheritance:

Spatial Transforms
******************

.. currentmodule:: rising.transforms.spatial

:hidden:`Mirror`
~~~~~~~~~~~~~~~~

.. autoclass:: Mirror
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`Rot90`
~~~~~~~~~~~~~~~

.. autoclass:: Rot90
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`Resize`
~~~~~~~~~~~~~~~~

.. autoclass:: Resize
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`Zoom`
~~~~~~~~~~~~~~

.. autoclass:: Zoom
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`ProgressiveResize`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ProgressiveResize
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`SizeStepScheduler`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SizeStepScheduler
    :members:
    :undoc-members:
    :show_inheritance:

Tensor Transforms
*****************
 .. py:currentmodule:: rising.transforms.tensor

:hidden:`ToTensor`
~~~~~~~~~~~~~~~~~~

.. autoclass:: ToTensor
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`ToDevice`
~~~~~~~~~~~~~~~~~~

.. autoclass:: ToDevice
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`TensorOp`
~~~~~~~~~~~~~~~~~~

.. autoclass:: TensorOp
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`Permute`
~~~~~~~~~~~~~~~~~

.. autoclass:: Permute
    :members:
    :undoc-members:
    :show_inheritance:

Utility Transforms
******************

.. currentmodule:: rising.transforms.utility

:hidden:`DoNothing`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: DoNothing
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`SegToBox`
~~~~~~~~~~~~~~~~~~

.. autoclass:: SegToBox
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`BoxToSeg`
~~~~~~~~~~~~~~~~~~

.. autoclass:: BoxToSeg
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`InstanceToSemantic`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: InstanceToSemantic
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`PopKeys`
~~~~~~~~~~~~~~~~~

.. autoclass:: PopKeys
    :members:
    :undoc-members:
    :show_inheritance:

:hidden:`FilterKeys`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FilterKeys
    :members:
    :undoc-members:
    :show_inheritance:

Functional Interface
--------------------

.. automodule:: rising.transforms.functional