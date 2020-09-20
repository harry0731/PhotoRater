//
//  Photorater.swift
//  StyleTransfer
//
//  Created by hugo on 2020/9/5.
//  Copyright © 2020 TensorFlow Authors. All rights reserved.
//

import TensorFlowLite
import UIKit

class PhotoRater {
    
    private var predictInterpreter: Interpreter
    
    private let tfLiteQueue: DispatchQueue
    
    /// Create a Style Transferer instance with a quantized Int8 model that runs inference on the CPU.
    static func newCPUStyleTransferer(
      completion: @escaping ((Result<PhotoRater>) -> Void)
    ) -> () {
      return PhotoRater.newInstance(
        useMetalDelegate: false,
        completion: completion)
    }

    static func newGPUStyleTransferer(
      completion: @escaping ((Result<PhotoRater>) -> Void)
    ) -> () {
      return PhotoRater.newInstance(
        useMetalDelegate: true,
        completion: completion)
    }
    
    static func newInstance(
        useMetalDelegate: Bool,
        completion: @escaping ((Result<PhotoRater>) -> Void)) {
    // Create a dispatch queue to ensure all operations on the Intepreter will run serially.
    let tfLiteQueue = DispatchQueue(label: "photorater")

    // Run initialization in background thread to avoid UI freeze.
    tfLiteQueue.async {
      // Construct the path to the model file.
      guard
          let predictModelPath = Bundle.main.path(
            forResource: Constants.Float16.predictModel,
            ofType: Constants.modelFileExtension
          )
      else {
        completion(.error(InitializationError.invalidModel(
          "model could not be loaded"
        )))
        return
      }

      // Specify the delegate for the TF Lite `Interpreter`.
      let createDelegates: () -> [Delegate]? = {
        if useMetalDelegate {
          return [MetalDelegate()]
        }
        return nil
      }
      let createOptions: () -> Interpreter.Options? = {
        if useMetalDelegate {
          return nil
        }
        var options = Interpreter.Options()
        options.threadCount = ProcessInfo.processInfo.processorCount >= 2 ? 2 : 1
        return options
      }

      do {
        // Create the `Interpreter`s.
        let predictInterpreter = try Interpreter(
          modelPath: predictModelPath,
          options: createOptions(),
          delegates: createDelegates()
        )

        // Allocate memory for the model's input `Tensor`s.
        try predictInterpreter.allocateTensors()

        // Create an StyleTransferer instance and return.
        let photorater = PhotoRater(
          tfLiteQueue: tfLiteQueue,
          predictInterpreter: predictInterpreter
        )
        DispatchQueue.main.async {
          completion(.success(photorater))
        }
      } catch let error {
        print("Failed to create the interpreter with error: \(error.localizedDescription)")
        DispatchQueue.main.async {
          completion(.error(InitializationError.internalError(error)))
        }
        return
        }
      }
    }
    /// Initialize Style Transferer instance.
    fileprivate init(
      tfLiteQueue: DispatchQueue,
      predictInterpreter: Interpreter
    ) {
      // Store TF Lite intepreter
      self.predictInterpreter = predictInterpreter

      // Store the dedicated DispatchQueue for TFLite.
      self.tfLiteQueue = tfLiteQueue
    }
    
    
    func runStyleTransfer(
                        photo image: UIImage,
                        completion: @escaping ((Result<PhotoRaterResult>) -> Void)) {
    tfLiteQueue.async {
      let outputTensor: Tensor
//      let startTime: Date = Date()
//      var preprocessingTime: TimeInterval = 0
//      var stylePredictTime: TimeInterval = 0
//      var styleTransferTime: TimeInterval = 0
//      var postprocessingTime: TimeInterval = 0

//      func timeSinceStart() -> TimeInterval {
//        return abs(startTime.timeIntervalSinceNow)
//      }
        

      do {
        guard
          let inputRGBData = image.scaledData(
            with: CGSize(width: 224, height: 224),
            isQuantized: false
          )
        else {
          DispatchQueue.main.async {
            completion(.error(StyleTransferError.invalidImage))
          }
          print("Failed to convert the input image buffer to RGB data.")
          return
        }

//        preprocessingTime = timeSinceStart()

        // Copy the RGB data to the input `Tensor`.
        try self.predictInterpreter.copy(inputRGBData, toInputAt: 0)

        // Run inference by invoking the `Interpreter`.
        try self.predictInterpreter.invoke()

        // Get the output `Tensor` to process the inference results.
        outputTensor = try self.predictInterpreter.output(at: 0)

        // Grab bottleneck data from output tensor.

//        stylePredictTime = timeSinceStart() - preprocessingTime

        // Copy the RGB and bottleneck data to the input `Tensor`.
//        try self.transferInterpreter.copy(inputRGBData, toInputAt: 0)
//        try self.transferInterpreter.copy(bottleneck, toInputAt: 1)

        // Run inference by invoking the `Interpreter`.
//        try self.transferInterpreter.invoke()

        // Get the result tensor
//        outputTensor = try self.transferInterpreter.output(at: 0)

//        styleTransferTime = timeSinceStart() - stylePredictTime - preprocessingTime

      } catch let error {
        print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
        DispatchQueue.main.async {
          completion(.error(StyleTransferError.internalError(error)))
        }
        return
      }
        
        let result: Float
        result = outputTensor.data.toArray(type: Float32.self)[0]
        
        print(result)

      // Construct image from output tensor data
//      guard let cgImage = self.postprocessImageData(data: outputTensor.data) else {
//        DispatchQueue.main.async {
//          completion(.error(StyleTransferError.resultVisualizationError))
//        }
//        return
//      }
//
//      let outputImage = UIImage(cgImage: cgImage)

//      postprocessingTime =
//          timeSinceStart() - stylePredictTime - styleTransferTime - preprocessingTime

      // Return the result.
      DispatchQueue.main.async {
        completion(
          .success(
            PhotoRaterResult(
              result: result
            )
          )
        )
      }
    }
  }
}

/// Convenient enum to return result with a callback
enum Result<T> {
  case success(T)
  case error(Error)
}

/// Define errors that could happen in the initialization of this class
enum InitializationError: Error {
  // Invalid TF Lite model
  case invalidModel(String)

  // Invalid label list
  case invalidLabelList(String)

  // TF Lite Internal Error when initializing
  case internalError(Error)
}

/// Define errors that could happen when running style transfer
enum StyleTransferError: Error {
  // Invalid input image
  case invalidImage

  // TF Lite Internal Error when initializing
  case internalError(Error)

  // Invalid input image
  case resultVisualizationError
}

struct PhotoRaterResult {
    let result: Float
}

private enum Constants {

  // Namespace for Float16 models, optimized for GPU inference.
  enum Float16 {

    static let predictModel = "predict"

  }

  static let modelFileExtension = "tflite"

  static let styleImageSize = CGSize(width: 224, height: 224)

  static let inputImageSize = CGSize(width: 224, height: 224)

}
