//
//  UIImageExtension.swift
//  HairSegmentation
//
//  Created by Артем Шафаростов on 23.07.2020.
//

import UIKit

public struct PixelData {
    var a: UInt8
    var r: UInt8
    var g: UInt8
    var b: UInt8
}
/// Helper functions for the UIImage class that is useful for this sample app.
extension UIImage {
  convenience init?(pixels: [PixelData], width: Int, height: Int) {
    guard width > 0 && height > 0, pixels.count == width * height else { return nil }
    var data = pixels
    guard let providerRef = CGDataProvider(data: Data(bytes: &data, count: data.count * MemoryLayout<PixelData>.size) as CFData)
        else { return nil }
    guard let cgim = CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        bytesPerRow: width * MemoryLayout<PixelData>.size,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue),
        provider: providerRef,
        decode: nil,
        shouldInterpolate: true,
        intent: .defaultIntent)
    else { return nil }
    self.init(cgImage: cgim)
  }
  /// Helper function to center-crop image.
  /// - Returns: Center-cropped copy of this image
  func cropCenter() -> UIImage? {
    let isPortrait = size.height > size.width
    let isLandscape = size.width > size.height
    let breadth = min(size.width, size.height)
    let breadthSize = CGSize(width: breadth, height: breadth)
    let breadthRect = CGRect(origin: .zero, size: breadthSize)

    UIGraphicsBeginImageContextWithOptions(breadthSize, false, scale)
    let croppingOrigin = CGPoint(
      x: isLandscape ? floor((size.width - size.height) / 2) : 0,
      y: isPortrait ? floor((size.height - size.width) / 2) : 0
    )
    guard let cgImage = cgImage?.cropping(to: CGRect(origin: croppingOrigin, size: breadthSize))
    else { return nil }
    UIImage(cgImage: cgImage).draw(in: breadthRect)
    let croppedImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()

    return croppedImage
  }

  /// Overlay an image on top of current image with alpha component
  /// - Parameters
  ///   - alpha: Alpha component of the image to be drawn on the top of current image
  /// - Returns: The overlayed image or `nil` if the image could not be drawn.
  func overlayWithImage(image: UIImage, alpha: Float) -> UIImage? {
    let areaSize = CGRect(x: 0, y: 0, width: self.size.width, height: self.size.height)

    UIGraphicsBeginImageContext(self.size)
    self.draw(in: areaSize)
    image.draw(in: areaSize, blendMode: .normal, alpha: CGFloat(alpha))
    let newImage: UIImage? = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()

    return newImage
  }
    
  func scalePreservingAspectRatio(targetSize: CGSize) -> UIImage {
            // Determine the scale factor that preserves aspect ratio
    let widthRatio = targetSize.width / size.width
    let heightRatio = targetSize.height / size.height
    
    let scaleFactor = min(widthRatio, heightRatio)
    
    // Compute the new image size that preserves aspect ratio
    let scaledImageSize = CGSize(
        width: size.width * scaleFactor,
        height: size.height * scaleFactor
    )

    // Draw and return the resized UIImage
    let renderer = UIGraphicsImageRenderer(
        size: scaledImageSize
    )

    let scaledImage = renderer.image { _ in
        self.draw(in: CGRect(
            origin: .zero,
            size: scaledImageSize
        ))
    }
    
    return scaledImage
  }
  func pixelData() -> [UInt8]? {
    let size = self.size
    let dataSize = size.width * size.height * 4
    var pixelData = [UInt8](repeating: 0, count: Int(dataSize))
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(data: &pixelData,
                            width: Int(size.width),
                            height: Int(size.height),
                            bitsPerComponent: 8,
                            bytesPerRow: 4 * Int(size.width),
                            space: colorSpace,
                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    guard let cgImage = self.cgImage else { return nil }
    context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))

    return pixelData
  }
}

/// Helper functions for the UIKit class that is useful for this sample app.
extension UIColor {

  // Check if the color is light or dark, as defined by the injected lightness threshold.
  // A nil value is returned if the lightness couldn't be determined.
  func isLight(threshold: Float = 0.5) -> Bool? {
    let originalCGColor = self.cgColor

    // Convert the color to the RGB colorspace as some color such as UIColor.white and .black
    // are grayscale.
    let RGBCGColor = originalCGColor.converted(
      to: CGColorSpaceCreateDeviceRGB(), intent: .defaultIntent, options: nil)

    guard let components = RGBCGColor?.components else { return nil }
    guard components.count >= 3 else { return nil }

    // Calculate color brightness according to Digital ITU BT.601.
    let brightness = Float(
      ((components[0] * 299) + (components[1] * 587) + (components[2] * 114)) / 1000
    )

    return (brightness > threshold)
  }
}
