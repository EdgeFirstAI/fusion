extern crate num;

use crate::{
    tensorflowlite_c, TfLiteError, TfLiteTensor, TfLiteType_kTfLiteBFloat16,
    TfLiteType_kTfLiteBool, TfLiteType_kTfLiteComplex128, TfLiteType_kTfLiteComplex64,
    TfLiteType_kTfLiteFloat16, TfLiteType_kTfLiteFloat32, TfLiteType_kTfLiteFloat64,
    TfLiteType_kTfLiteInt16, TfLiteType_kTfLiteInt32, TfLiteType_kTfLiteInt4,
    TfLiteType_kTfLiteInt64, TfLiteType_kTfLiteInt8, TfLiteType_kTfLiteNoType,
    TfLiteType_kTfLiteResource, TfLiteType_kTfLiteString, TfLiteType_kTfLiteUInt16,
    TfLiteType_kTfLiteUInt32, TfLiteType_kTfLiteUInt64, TfLiteType_kTfLiteUInt8,
    TfLiteType_kTfLiteVariant,
};

pub use crate::TfLiteQuantizationParams;
use std::{ffi::CStr, ptr};

#[derive(Copy, Clone, PartialEq, Eq, FromPrimitive, Debug)]
pub enum TensorType {
    UnknownType = -1,
    NoType = TfLiteType_kTfLiteNoType as isize,
    Float32 = TfLiteType_kTfLiteFloat32 as isize,
    Int32 = TfLiteType_kTfLiteInt32 as isize,
    UInt8 = TfLiteType_kTfLiteUInt8 as isize,
    Int64 = TfLiteType_kTfLiteInt64 as isize,
    String = TfLiteType_kTfLiteString as isize,
    Bool = TfLiteType_kTfLiteBool as isize,
    Int16 = TfLiteType_kTfLiteInt16 as isize,
    Complex64 = TfLiteType_kTfLiteComplex64 as isize,
    Int8 = TfLiteType_kTfLiteInt8 as isize,
    Float16 = TfLiteType_kTfLiteFloat16 as isize,
    Float64 = TfLiteType_kTfLiteFloat64 as isize,
    Complex128 = TfLiteType_kTfLiteComplex128 as isize,
    UInt64 = TfLiteType_kTfLiteUInt64 as isize,
    Resource = TfLiteType_kTfLiteResource as isize,
    Variant = TfLiteType_kTfLiteVariant as isize,
    UInt32 = TfLiteType_kTfLiteUInt32 as isize,
    UInt16 = TfLiteType_kTfLiteUInt16 as isize,
    Int4 = TfLiteType_kTfLiteInt4 as isize,
    BFloat16 = TfLiteType_kTfLiteBFloat16 as isize,
}

pub struct TensorMut<'a> {
    pub(crate) ptr: ptr::NonNull<TfLiteTensor>,
    pub(crate) lib: &'a tensorflowlite_c,
}

impl<'a> TensorMut<'a> {
    pub fn tensor_type(&self) -> TensorType {
        let ret = unsafe { self.lib.TfLiteTensorType(self.ptr.as_ptr()) };
        match num::FromPrimitive::from_u32(ret) {
            Some(v) => v,
            None => TensorType::UnknownType,
        }
    }

    pub fn num_dims(&self) -> Result<usize, TfLiteError> {
        match usize::try_from(unsafe { self.lib.TfLiteTensorNumDims(self.ptr.as_ptr()) }) {
            Ok(v) => Ok(v),
            Err(_) => Err(TfLiteError::new(format!(
                "Tensor {} does not have dims set",
                self.name()
            ))), /* returned -1 because dims not set , */
        }
    }

    pub fn dim(&self, i: usize) -> Result<usize, TfLiteError> {
        let num_dims = self.num_dims()?;
        if i >= num_dims {
            return Err(TfLiteError::new(format!(
                "Tried to access dim {} of {}",
                i, num_dims
            )));
        }
        let i = i32::try_from(i).unwrap();
        Ok(usize::try_from(unsafe { self.lib.TfLiteTensorDim(self.ptr.as_ptr(), i) }).unwrap())
    }

    pub fn byte_size(&self) -> usize {
        unsafe { self.lib.TfLiteTensorByteSize(self.ptr.as_ptr()) }
    }

    pub fn name(&self) -> &str {
        unsafe { CStr::from_ptr(self.lib.TfLiteTensorName(self.ptr.as_ptr())) }
            .to_str()
            .unwrap()
    }

    pub fn shape(&self) -> Result<Vec<usize>, TfLiteError> {
        let num_dims = self.num_dims()?;
        let mut dims = Vec::with_capacity(num_dims);
        for i in 0..num_dims {
            dims.push(self.dim(i)?);
        }
        Ok(dims)
    }

    pub fn volume(&self) -> Result<usize, TfLiteError> {
        Ok(self.shape()?.iter().fold(1, |acc, x| acc * x))
    }

    pub fn maprw<'b, T>(&'b mut self) -> Result<&'b mut [T], TfLiteError> {
        let volume = self.volume()?;
        if std::mem::size_of::<T>() * volume > self.byte_size() {
            return Err(TfLiteError::new(format!(
                "Tensor too small to map as {}",
                std::any::type_name::<T>()
            )));
        }
        let ptr = unsafe { self.lib.TfLiteTensorData(self.ptr.as_ptr()) } as *mut T;
        if ptr.is_null() {
            return Err(TfLiteError::new("Tensor data is NULL"));
        }
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr, volume as usize) })
    }

    pub fn mapro<'b, T>(&'b self) -> Result<&'b [T], TfLiteError> {
        let volume = self.volume()?;
        if std::mem::size_of::<T>() * volume > self.byte_size() {
            return Err(TfLiteError::new(format!(
                "Tensor too small to map as {}",
                std::any::type_name::<T>()
            )));
        }
        let ptr = unsafe { self.lib.TfLiteTensorData(self.ptr.as_ptr()) } as *mut T;
        if ptr.is_null() {
            return Err(TfLiteError::new("Tensor data is NULL"));
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr, volume as usize) })
    }

    pub fn get_quantization_params(&self) -> TfLiteQuantizationParams {
        unsafe { self.lib.TfLiteTensorQuantizationParams(self.ptr.as_ptr()) }
    }
}

pub struct Tensor<'a> {
    pub(crate) ptr: *const TfLiteTensor,
    pub(crate) lib: &'a tensorflowlite_c,
}

impl<'a> Tensor<'a> {
    pub fn tensor_type(&self) -> TensorType {
        let ret = unsafe { self.lib.TfLiteTensorType(self.ptr) };
        match num::FromPrimitive::from_u32(ret) {
            Some(v) => v,
            None => TensorType::UnknownType,
        }
    }

    pub fn num_dims(&self) -> Result<usize, TfLiteError> {
        match usize::try_from(unsafe { self.lib.TfLiteTensorNumDims(self.ptr) }) {
            Ok(v) => Ok(v),
            Err(_) => Err(TfLiteError::new(format!(
                "Tensor {} does not have dims set",
                self.name()
            ))), /* returned -1 because dims not set , */
        }
    }

    pub fn dim(&self, i: usize) -> Result<usize, TfLiteError> {
        let num_dims = self.num_dims()?;
        if i >= num_dims {
            return Err(TfLiteError::new(format!(
                "Tried to access dim {} of {}",
                i, num_dims
            )));
        }
        let i = i32::try_from(i).unwrap();
        Ok(usize::try_from(unsafe { self.lib.TfLiteTensorDim(self.ptr, i) }).unwrap())
    }

    pub fn byte_size(&self) -> usize {
        unsafe { self.lib.TfLiteTensorByteSize(self.ptr) }
    }

    pub fn name(&self) -> &str {
        unsafe { CStr::from_ptr(self.lib.TfLiteTensorName(self.ptr)) }
            .to_str()
            .unwrap()
    }

    pub fn shape(&self) -> Result<Vec<usize>, TfLiteError> {
        let num_dims = self.num_dims()?;
        let mut dims = Vec::with_capacity(num_dims);
        for i in 0..num_dims {
            dims.push(self.dim(i)?);
        }
        Ok(dims)
    }

    pub fn volume(&self) -> Result<usize, TfLiteError> {
        Ok(self.shape()?.iter().fold(1, |acc, x| acc * x))
    }

    pub fn mapro<'b, T>(&'b self) -> Result<&'b [T], TfLiteError> {
        let volume = self.volume()?;
        if std::mem::size_of::<T>() * volume > self.byte_size() {
            return Err(TfLiteError::new(format!(
                "Tensor too small to map as {}",
                std::any::type_name::<T>()
            )));
        }
        let ptr = unsafe { self.lib.TfLiteTensorData(self.ptr) } as *mut T;
        if ptr.is_null() {
            return Err(TfLiteError::new("Tensor data is NULL"));
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr, volume as usize) })
    }

    pub fn get_quantization_params(&self) -> TfLiteQuantizationParams {
        unsafe { self.lib.TfLiteTensorQuantizationParams(self.ptr) }
    }
}

impl<'a> std::fmt::Debug for Tensor<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dims = self.num_dims().unwrap_or(0);
        let mut first = true;
        write!(f, "{}: ", self.name())?;
        for i in 0..dims {
            if !first {
                f.write_str("x")?;
            }
            first = false;
            write!(f, "{}", self.dim(i).unwrap_or(0))?;
        }
        write!(f, " {:?}", self.tensor_type())
    }
}

impl<'a> std::fmt::Debug for TensorMut<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dims = self.num_dims().unwrap_or(0);
        let mut first = true;
        write!(f, "{}: ", self.name())?;
        for i in 0..dims {
            if !first {
                f.write_str("x")?;
            }
            first = false;
            write!(f, "{}", self.dim(i).unwrap_or(0))?;
        }
        write!(f, " {:?}", self.tensor_type())
    }
}
