.. role:: hidden
    :class: hidden-section

rising.transforms
========================================

.. automodule:: rising.transforms

.. currentmodule:: rising.transforms

Transformation Base Classes
***************************

.. automodule:: rising.transforms.abstract

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

Compose Transforms
******************

.. automodule:: rising.transforms.compose

.. currentmodule:: rising.transforms.compose

:hidden:`Compose`
~~~~~~~~~~~~~~~~~

.. autoclass:: Compose
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`DropoutCompose`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DropoutCompose
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`dict_call`
~~~~~~~~~~~~~~~~~~~

.. autofunction:: dict_call

Affine Transforms
*****************

.. automodule:: rising.transforms.affine

.. currentmodule:: rising.transforms.affine

:hidden:`Affine`
~~~~~~~~~~~~~~~~

.. autoclass:: Affine
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`StackedAffine`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: StackedAffine
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`BaseAffine`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BaseAffine
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`Rotate`
~~~~~~~~~~~~~~~~~

.. autoclass:: Rotate
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`Translate`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Translate
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`Scale`
~~~~~~~~~~~~~~~~

.. autoclass:: Scale
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`Resize`
~~~~~~~~~~~~~~~~~

.. autoclass:: Resize
    :members:
    :undoc-members:
    :show-inheritance:

Channel Transforms
*******************

.. automodule:: rising.transforms.channel

.. currentmodule:: rising.transforms.channel

:hidden:`OneHot`
~~~~~~~~~~~~~~~~~

.. autoclass:: OneHot
    :members:
    :undoc-members:
    :show-inheritance:

Cropping Transforms
********************

.. automodule:: rising.transforms.crop

.. currentmodule:: rising.transforms.crop

:hidden:`CenterCrop`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CenterCrop
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`RandomCrop`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomCrop
    :members:
    :undoc-members:
    :show-inheritance:

Format Transforms
******************

.. automodule:: rising.transforms.format

.. currentmodule:: rising.transforms.format

:hidden:`MapToSeq`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: MapToSeq
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`SeqToMap`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: SeqToMap
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`PopKeys`
~~~~~~~~~~~~~~~~~~

.. autoclass:: PopKeys
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`FilterKeys`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FilterKeys
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`RenameKeys`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RenameKeys
    :members:
    :undoc-members:
    :show-inheritance:

Intensity Transforms
*********************

.. automodule:: rising.transforms.intensity

.. currentmodule:: rising.transforms.intensity

:hidden:`Clamp`
~~~~~~~~~~~~~~~~

.. autoclass:: Clamp
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`NormRange`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NormRange
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`NormMinMax`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NormMinMax
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`NormZeroMeanUnitStd`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NormZeroMeanUnitStd
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`NormMeanStd`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NormMeanStd
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`Noise`
~~~~~~~~~~~~~~~~

.. autoclass:: Noise
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`GaussianNoise`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GaussianNoise
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`ExponentialNoise`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ExponentialNoise
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`GammaCorrection`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GammaCorrection
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`RandomValuePerChannel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomValuePerChannel
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`RandomAddValue`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomAddValue
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`RandomScaleValue`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomScaleValue
    :members:
    :undoc-members:
    :show-inheritance:

Kernel Transforms
******************

.. automodule:: rising.transforms.kernel

.. currentmodule:: rising.transforms.kernel

:hidden:`KernelTransform`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: KernelTransform
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`GaussianSmoothing`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GaussianSmoothing
    :members:
    :undoc-members:
    :show-inheritance:

Spatial Transforms
*******************

.. automodule:: rising.transforms.spatial

.. currentmodule:: rising.transforms.spatial

:hidden:`Mirror`
~~~~~~~~~~~~~~~~~

.. autoclass:: Mirror
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`Rot90`
~~~~~~~~~~~~~~~~

.. autoclass:: Rot90
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`ResizeNative`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ResizeNative
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`Zoom`
~~~~~~~~~~~~~~~

.. autoclass:: Zoom
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`ProgressiveResize`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ProgressiveResize
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`SizeStepScheduler`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SizeStepScheduler
    :members:
    :undoc-members:
    :show-inheritance:

Tensor Transforms
******************

.. automodule:: rising.transforms.tensor

.. currentmodule:: rising.transforms.tensor

:hidden:`ToTensor`
~~~~~~~~~~~~~~~~~~

.. autoclass:: ToTensor
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`ToDeviceDtype`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ToDeviceDtype
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`ToDevice`
~~~~~~~~~~~~~~~~~~

.. autoclass:: ToDevice
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`ToDtype`
~~~~~~~~~~~~~~~~~

.. autoclass:: ToDtype
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`TensorOp`
~~~~~~~~~~~~~~~~~~

.. autoclass:: TensorOp
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`Permute`
~~~~~~~~~~~~~~~~~

.. autoclass:: Permute
    :members:
    :undoc-members:
    :show-inheritance:

Utility Transforms
*******************

.. automodule:: rising.transforms.utility

.. currentmodule:: rising.transforms.utility

:hidden:`DoNothing`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: DoNothing
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`SegToBox`
~~~~~~~~~~~~~~~~~~

.. autoclass:: SegToBox
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`BoxToSeg`
~~~~~~~~~~~~~~~~~~

.. autoclass:: BoxToSeg
    :members:
    :undoc-members:
    :show-inheritance:

:hidden:`InstanceToSemantic`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: InstanceToSemantic
    :members:
    :undoc-members:
    :show-inheritance:
