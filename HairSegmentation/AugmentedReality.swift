//
//  AugmentedReality.swift
//  HairSegmentation
//
//  Created by Артем Шафаростов on 29.07.2020.
//

import ARKit
import RealityKit
import SwiftUI
import Vision

struct ARContentView: View {
    var body: some View {
        ARViewContainer()
//            .edgesIgnoringSafeArea(.all)
//            .frame(height: 700)
    }
}


struct ARViewContainer: UIViewRepresentable {
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    func makeUIView(context: Context) -> ARView {
        context.coordinator.arView.session.delegate = context.coordinator
        
        return context.coordinator.arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {}
    
    class Coordinator: NSObject, ARSessionDelegate {
        var parent: ARViewContainer
        var arView: ARView
        var configuration: ARImageTrackingConfiguration?
        var image: UIImage?
        let Context: CIContext
        
        init(_ arViewContainer: ARViewContainer) {
            self.parent = arViewContainer
            // Init AR
            self.arView = ARView(frame: .zero,
                                 cameraMode: ARView.CameraMode.ar,
                                 automaticallyConfigureSession: true)
            self.Context = CIContext()
        }
        
        func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
            
        }
        
        func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
            
        }
        
        func session(_ session: ARSession, didUpdate frame: ARFrame) {
            let image = CIImage(cvPixelBuffer: frame.capturedImage)
            let filter = CIFilter(name: "CIPixellate")!
            filter.setValue(image, forKey: kCIInputImageKey)
            let result = filter.outputImage!
            //self.arView.environment.background
            
            self.arView.environment.background = self.Context.createCGImage(result, from: image.extent)
            self.arView.environment.background.contentsTransform = SCNMatrix4MakeRotation(.pi / 2, 0, 0, 1)
        }
    }

}
