//
//  PostProcessing.swift
//  HairSegmentation
//
//  Created by Артем Шафаростов on 22.07.2020.
//

import SwiftUI
import UIKit

struct PostProcessing: View {
    @State private var inference: InferenceImage?
    @ObservedObject var currentImage = CurrentImage()
    
    var body: some View {
        VStack {
            Image(uiImage: currentImage.image).resizable()
                .frame(width: 224, height: 224)
            Text("Hi, Max!")
        }.onAppear {
            self.initInference()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                // Put your code which should be executed with a delay here
                let image = readImage(img_name: "maxim")
                print("INIT - ", self.inference?.initialized)
                self.inference?.runSegmentation(image)
            }
            let timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { timer in
                if self.inference?.segmentationResult != nil {
                    //let image = UIImage(pixels: pixels, width: width, height: height)
                    self.currentImage.image = (self.inference?.segmentationResult?.overlayImage as UIImage?)!
                    self.currentImage.time = self.inference?.segmentationResult?.inferenceTime as! Double
                    //currentImage.image = image!
                    print("DEBUG: TIME - ", self.currentImage.time)
                    timer.invalidate()
                }
            }
        }
    }
    
    func initInference() {
        self.inference = InferenceImage()
        self.inference?.setSegmentator()
    }
    
    func showImage() -> UIImage {
        print("YES2")
        let image = readImage(img_name: "maxim")
//        repeat {
//            self.inference?.runSegmentation(image)
//        } while self.inference?.segmentationResult == nil
        // self.inference?.runSegmentation(image)
        //print("DEBUG: RESULT - ", self.inference?.segmentationResult)
        return image
    }
}


class CurrentImage: ObservableObject {
    @Published var image = readImage(img_name: "maxim")
    @Published var time = 0.0
}

func readImage(img_name: String) -> UIImage {
    let image = UIImage(named: img_name) as! UIImage
//    let inference = InferenceImage()
//    inference.setSegmentator()
//    // let inputImage = readImage(img_name: "photo")
//    print(image)
//    inference.runSegmentation(image)
//    print("PROCESSED - RUN SEGMENTATION")
//    print("DEBUG segmentationResult - ,", inference.segmentationResult)
    return image
}

struct PostProcessing_Previews: PreviewProvider {
    static var previews: some View {
        PostProcessing()
    }
}


class InferenceImage {
    private var imageSegmentator: ImageSegmentator?
    public var segmentationResult: SegmentationResult?
    public var initialized: Bool = false
    
    func setSegmentator() {
        ImageSegmentator.newInstance { result in
            switch result {
                case let .success(segmentator):
                    self.imageSegmentator = segmentator
                    self.initialized = true
                case .error(_):
                    print("Failed to initialize.")
            }
        }
    }
    
    func runSegmentation(_ image: UIImage) {
        print("DEBUG: RUN SEGMENTATION")
        print("DEBUG: IMAGE - ", image)
        print("segmentator - ", self.imageSegmentator)
        self.imageSegmentator?.runSegmentation(
            image,
            completion: { result in
                switch result {
                case let .success(segmentationResult):
                    self.segmentationResult = segmentationResult
                case let .error(error):
                    print("Everything was wrong, Dude!")
                }
            })
        print("SHIT")
    }
}
