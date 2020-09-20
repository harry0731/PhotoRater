// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import UIKit
import os

class ViewController: UIViewController {

  /// Image picker for accessing the photo library or camera.
  private var imagePicker = UIImagePickerController()

  /// Style transferer instance reponsible for running the TF model. Uses a Int8-based model and
  /// runs inference on the CPU.
//  private var cpuStyleTransferer: StyleTransferer?
  private var cpuStyleTransferer: PhotoRater?

  /// Style transferer instance reponsible for running the TF model. Uses a Float16-based model and
  /// runs inference on the GPU.
//  private var gpuStyleTransferer: StyleTransferer?
  private var gpuStyleTransferer: PhotoRater?


  /// Target image to transfer a style onto.
  private var targetImage: UIImage?

  /// Style-representative image applied to the input image to create a pastiche.
  private var styleImage: UIImage?

  /// Style transfer result.
//  private var styleTransferResult: StyleTransferResult?
  private var styleTransferResult: PhotoRaterResult?

  // UI elements
  @IBOutlet weak var imageView: UIImageView!
  @IBOutlet weak var photoCameraButton: UIButton!
  @IBOutlet weak var inferenceStatusLabel: UILabel!
  @IBOutlet weak var styleImageView: UIImageView!
  @IBOutlet weak var runButton: UIButton!

  override func viewDidLoad() {
    super.viewDidLoad()

    imageView.contentMode = .scaleAspectFill

    // Setup image picker.
    imagePicker.delegate = self
    imagePicker.sourceType = .photoLibrary

    // Enable camera option only if current device has camera.
    let isCameraAvailable = UIImagePickerController.isCameraDeviceAvailable(.front)
      || UIImagePickerController.isCameraDeviceAvailable(.rear)
    if isCameraAvailable {
      photoCameraButton.isEnabled = true
    }


    // Initialize new style transferer instances.
    PhotoRater.newCPUStyleTransferer { result in
      switch result {
      case .success(let transferer):
        self.cpuStyleTransferer = transferer
      case .error(let wrappedError):
        print("Failed to initialize: \(wrappedError)")
      }
    }
    PhotoRater.newGPUStyleTransferer { result in
      switch result {
      case .success(let transferer):
        self.gpuStyleTransferer = transferer
      case .error(let wrappedError):
        print("Failed to initialize: \(wrappedError)")
      }
    }
  }

  override func viewDidAppear(_ animated: Bool) {
    super.viewDidAppear(animated)
  }

  override func viewDidDisappear(_ animated: Bool) {
    super.viewDidDisappear(animated)
    NotificationCenter.default.removeObserver(self)
  }

  @IBAction func onTapRunButton(_ sender: Any) {
    // Make sure that the cached target image is available.
    guard targetImage != nil else {
      self.inferenceStatusLabel.text = "Error: Input image is nil."
      return
    }

    runStyleTransfer(targetImage!)
  }

  /// Open camera to allow user taking photo.
  @IBAction func onTapOpenCamera(_ sender: Any) {
    guard
      UIImagePickerController.isCameraDeviceAvailable(.front)
        || UIImagePickerController.isCameraDeviceAvailable(.rear)
    else {
      return
    }

    imagePicker.sourceType = .camera
    present(imagePicker, animated: true)
  }

  /// Open photo library for user to choose an image from.
  @IBAction func onTapPhotoLibrary(_ sender: Any) {
    imagePicker.sourceType = .photoLibrary
    present(imagePicker, animated: true)
  }

}

// MARK: - Style Transfer

extension ViewController {
  /// Run style transfer on the given image, and show result on screen.
  ///  - Parameter image: The target image for style transfer.
  func runStyleTransfer(_ image: UIImage) {
    clearResults()

    let shouldUseQuantizedFloat16 = true
    let transferer = shouldUseQuantizedFloat16 ? gpuStyleTransferer : cpuStyleTransferer

    // Make sure that the style transferer is initialized.
    guard let styleTransferer = transferer else {
      inferenceStatusLabel.text = "ERROR: Interpreter is not ready."
      return
    }

    guard let targetImage = self.targetImage else {
      inferenceStatusLabel.text = "ERROR: Select a target image."
      return
    }

    // Center-crop the target image if the user has enabled the option.
    let image = targetImage.cropCenter()

    // Cache the potentially cropped image.
    self.targetImage = image

    // Show the potentially cropped image on screen.
    imageView.image = image

    // Make sure that the image is ready before running style transfer.
    guard image != nil else {
      inferenceStatusLabel.text = "ERROR: Image could not be cropped."
      return
    }

    // Run style transfer.
    styleTransferer.runStyleTransfer(
//      style: styleImage,
      photo: image!,
      completion: { result in
        // Show the result on screen
        switch result {
        case let .success(styleTransferResult):
//          self.styleTransferResult = styleTransferResult
            self.inferenceStatusLabel.text = "Score: \(styleTransferResult.result)"

          // Show result metadata
        case let .error(error):
          self.inferenceStatusLabel.text = error.localizedDescription
        }

        // Regardless of the result, re-enable switching between different display modes
        self.runButton.isEnabled = true
      })
  }

  /// Clear result from previous run to prepare for new style transfer.
  private func clearResults() {
    inferenceStatusLabel.text = "Running inference with TensorFlow Lite..."
  }
}

// MARK: - UIImagePickerControllerDelegate

extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {

  func imagePickerController(
    _ picker: UIImagePickerController,
    didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
  ) {

    if let pickedImage = info[.originalImage] as? UIImage {
      // Rotate target image to .up orientation to avoid potential orientation misalignment.
      guard let targetImage = pickedImage.transformOrientationToUp() else {
        inferenceStatusLabel.text = "ERROR: Image orientation couldn't be fixed."
        return
      }

      self.targetImage = targetImage

      if styleImage != nil {
        runStyleTransfer(targetImage)
      } else {
        imageView.image = targetImage
      }
    }

    dismiss(animated: true)
  }
}

// MARK: Pasteboard images

extension ViewController {

  fileprivate func imageFromPasteboard() -> UIImage? {
    return UIPasteboard.general.images?.first
  }

  fileprivate func imageRoleSelectionAlert(image: UIImage) -> UIAlertController {
    let controller = UIAlertController(title: "Paste Image",
                                       message: nil,
                                       preferredStyle: .actionSheet)
    controller.popoverPresentationController?.sourceView = view
    let setInputAction = UIAlertAction(title: "Set input image", style: .default) { _ in
      // Rotate target image to .up orientation to avoid potential orientation misalignment.
      guard let targetImage = image.transformOrientationToUp() else {
        self.inferenceStatusLabel.text = "ERROR: Image orientation couldn't be fixed."
        return
      }

      self.targetImage = targetImage
      self.imageView.image = targetImage
    }
    let setStyleAction = UIAlertAction(title: "Set style image", style: .default) { _ in
      guard let croppedImage = image.cropCenter() else {
        self.inferenceStatusLabel.text = "ERROR: Unable to crop style image."
        return
      }

      self.styleImage = croppedImage
      self.styleImageView.image = croppedImage
    }
    let cancelAction = UIAlertAction(title: "Cancel", style: .cancel) { _ in
      controller.dismiss(animated: true, completion: nil)
    }
    controller.addAction(setInputAction)
    controller.addAction(setStyleAction)
    controller.addAction(cancelAction)

    return controller
  }
}
