//
//  PostProcessing.swift
//  HairSegmentation
//
//  Created by Артем Шафаростов on 22.07.2020.
//

import SwiftUI
import UIKit

struct PostProcessing: View {
    var body: some View {
        Image(uiImage: postprocess(img_name: "photo"))
    }
}

func postprocess(img_name: String) -> UIImage {
    let image = UIImage(named: img_name) as! UIImage
    let cgImage = image.cgImage
    return image
}

struct PostProcessing_Previews: PreviewProvider {
    static var previews: some View {
        PostProcessing()
    }
}
