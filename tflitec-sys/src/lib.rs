#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]

use delegate::Delegate;
use log::debug;
use std::{os::raw::c_void, path::Path, ptr, time::Instant};
use tensor::{Tensor, TensorMut};
include!("ffi.rs");

pub mod delegate;
pub mod tensor;
#[macro_use]
extern crate num_derive;

pub fn tflite_status_to_result(status: TfLiteStatus) -> Result<(), String> {
    match status {
        TfLiteStatus_kTfLiteOk => Ok(()),
        TfLiteStatus_kTfLiteError => Err(format!("TfLiteRuntimeError: {status}")),
        TfLiteStatus_kTfLiteDelegateError => Err(format!("TfLiteDelegateError: {status}")),
        TfLiteStatus_kTfLiteApplicationError => Err(format!("TfLiteApplicationError: {status}")),
        TfLiteStatus_kTfLiteDelegateDataNotFound => {
            Err(format!("TfLiteDelegateDataNotFound: {status}"))
        }
        TfLiteStatus_kTfLiteDelegateDataWriteError => {
            Err(format!("TfLiteDelegateDataWriteError: {status}"))
        }
        TfLiteStatus_kTfLiteDelegateDataReadError => {
            Err(format!("TfLiteDelegateDataReadError: {status}"))
        }
        TfLiteStatus_kTfLiteUnresolvedOps => Err(format!("TfLiteUnresolvedOps: {status}")),
        TfLiteStatus_kTfLiteCancelled => Err(format!("TfLiteCancelled: {status}")),
        TfLiteStatus_kTfLiteOutputShapeNotKnown => {
            Err(format!("TfLiteOutputShapeNotKnown: {status}"))
        }
        _ => Err(format!("Unknown TfLite error: {status}")),
    }
}

pub struct TFLiteLib {
    lib: tensorflowlite_c,
}

pub static DEFAULT_TFLITEC_PATH: &str = "libtensorflowlite_c.so";
pub static DEFAULT_TFLITECPP_PATH: &str = "libtensorflow-lite.so";
impl TFLiteLib {
    pub fn new() -> Result<Self, libloading::Error> {
        // try a bunch of versions...
        // we don't know which specific version of tflite is installed so we try a bunch
        // Takes around 25ms to try to open 500 shared library files on the EVK
        for versions in (1..50).rev() {
            for patch in (0..10).rev() {
                if let Ok(tflite_lib) = TFLiteLib::new_with_path(format!(
                    "{DEFAULT_TFLITECPP_PATH}.2.{versions}.{patch}"
                )) {
                    debug!("Found TFLiteLib: {DEFAULT_TFLITECPP_PATH}.2.{versions}.{patch}");
                    return Ok(tflite_lib);
                }
            }
        }

        if let Ok(tflite_lib) = TFLiteLib::new_with_path(DEFAULT_TFLITEC_PATH) {
            return Ok(tflite_lib);
        }
        let tflite_lib = TFLiteLib::new_with_path(DEFAULT_TFLITECPP_PATH)?;
        Ok(tflite_lib)
    }

    pub fn new_with_path<P>(path: P) -> Result<Self, libloading::Error>
    where
        P: AsRef<Path>,
    {
        Ok(Self {
            lib: unsafe { tensorflowlite_c::new(path.as_ref().as_os_str())? },
        })
    }

    pub fn new_model_from_mem(&self, model: Vec<u8>) -> Result<Model, String> {
        let m = unsafe {
            self.lib
                .TfLiteModelCreate(model.as_ptr() as *const c_void, model.len())
        };
        Ok(Model {
            ptr: ptr::NonNull::new(m).ok_or("TfLiteModelCreate returned NULL")?,
            model_mem: model,
            lib: &self.lib,
        })
    }

    pub fn new_interpreter_builder(&self) -> Result<InterpreterBuilder, String> {
        Ok(InterpreterBuilder {
            options: ptr::NonNull::new(unsafe { self.lib.TfLiteInterpreterOptionsCreate() })
                .ok_or("TfLiteInterpreterOptionsCreate returned NULL")?,
            delegates: Vec::new(),
            lib: &self.lib,
        })
    }
}

#[allow(dead_code)]
pub struct Model<'a> {
    ptr: ptr::NonNull<TfLiteModel>,
    model_mem: Vec<u8>,
    lib: &'a tensorflowlite_c,
}

impl<'a> Drop for Model<'a> {
    fn drop(&mut self) {
        unsafe { self.lib.TfLiteModelDelete(self.ptr.as_ptr()) };
    }
}

pub struct InterpreterBuilder<'a> {
    options: ptr::NonNull<TfLiteInterpreterOptions>,
    delegates: Vec<Delegate>,
    lib: &'a tensorflowlite_c,
}

impl<'a> InterpreterBuilder<'a> {
    pub fn add_owned_delegate(&mut self, d: Delegate) {
        unsafe {
            self.lib
                .TfLiteInterpreterOptionsAddDelegate(self.options.as_ptr(), d.delegate.as_ptr())
        }
        self.delegates.push(d);
    }

    pub fn build(mut self, model: &Model) -> Result<Interpreter<'a>, String> {
        let interpreter = unsafe {
            self.lib
                .TfLiteInterpreterCreate(model.ptr.as_ptr(), self.options.as_ptr())
        };
        let interpreter = Interpreter {
            interpreter: ptr::NonNull::new(interpreter)
                .ok_or("TfLiteInterpreterCreate returned NULL")?,
            _delegates: std::mem::replace(&mut self.delegates, Vec::new()),
            lib: self.lib,
        };
        tflite_status_to_result(unsafe {
            self.lib
                .TfLiteInterpreterAllocateTensors(interpreter.interpreter.as_ptr())
        })?;
        Ok(interpreter)
    }
}
impl<'a> Drop for InterpreterBuilder<'a> {
    fn drop(&mut self) {
        unsafe {
            self.lib
                .TfLiteInterpreterOptionsDelete(self.options.as_ptr())
        };
    }
}
pub struct Interpreter<'a> {
    interpreter: ptr::NonNull<TfLiteInterpreter>,
    _delegates: Vec<Delegate>,
    lib: &'a tensorflowlite_c,
}

impl<'a> Interpreter<'a> {
    pub fn invoke(&mut self) -> Result<(), String> {
        let tflite_status = unsafe { self.lib.TfLiteInterpreterInvoke(self.interpreter.as_ptr()) };
        tflite_status_to_result(tflite_status)
    }

    pub fn inputs(&self) -> Result<Vec<Tensor<'a>>, String> {
        let len = unsafe {
            self.lib
                .TfLiteInterpreterGetInputTensorCount(self.interpreter.as_ptr())
        };
        let mut inputs = Vec::new();
        for i in 0..len {
            let input = unsafe {
                self.lib
                    .TfLiteInterpreterGetInputTensor(self.interpreter.as_ptr(), i as i32)
            };
            if input.is_null() {
                return Err("TfLiteInterpreterGetInputTensor returned NULL".to_string());
            }
            inputs.push(Tensor {
                ptr: input,
                lib: self.lib,
            });
        }
        Ok(inputs)
    }

    pub fn inputs_mut(&self) -> Result<Vec<TensorMut<'a>>, String> {
        let len = unsafe {
            self.lib
                .TfLiteInterpreterGetInputTensorCount(self.interpreter.as_ptr())
        };
        let mut inputs = Vec::new();
        for i in 0..len {
            let input = unsafe {
                self.lib
                    .TfLiteInterpreterGetInputTensor(self.interpreter.as_ptr(), i as i32)
            };
            inputs.push(TensorMut {
                ptr: ptr::NonNull::new(input)
                    .ok_or("TfLiteInterpreterGetInputTensor returned NULL")?,
                lib: self.lib,
            });
        }
        Ok(inputs)
    }

    pub fn outputs(&self) -> Result<Vec<Tensor>, String> {
        let len = unsafe {
            self.lib
                .TfLiteInterpreterGetOutputTensorCount(self.interpreter.as_ptr())
        };
        let mut outputs = Vec::new();
        for i in 0..len {
            let output = unsafe {
                self.lib
                    .TfLiteInterpreterGetOutputTensor(self.interpreter.as_ptr(), i as i32)
            };
            if output.is_null() {
                return Err("TfLiteInterpreterGetOutputTensor returned NULL".to_string());
            }
            outputs.push(Tensor {
                ptr: output,
                lib: self.lib,
            });
        }
        Ok(outputs)
    }
}

impl<'a> Drop for Interpreter<'a> {
    fn drop(&mut self) {
        unsafe { self.lib.TfLiteInterpreterDelete(self.interpreter.as_ptr()) };
    }
}
