//
//  HairSegmentationApp.swift
//  HairSegmentation
//
//  Created by Артем Шафаростов on 22.07.2020.
//

import UIKit
import SwiftUI
import TensorFlowLite

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?


    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {

        // Create the SwiftUI view that provides the window contents.
        //let contentView = PostProcessing()
        let contentView = ARContentView()
        print(contentView)
        // Use a UIHostingController as window root view controller.
        let window = UIWindow(frame: UIScreen.main.bounds)
        window.rootViewController = UIHostingController(rootView: contentView)
        self.window = window
        window.makeKeyAndVisible()
        return true
    }
}
