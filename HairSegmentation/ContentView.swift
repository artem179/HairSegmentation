//
//  ContentView.swift
//  HairSegmentation
//
//  Created by Артем Шафаростов on 22.07.2020.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        Image("photo").resizable()
            .frame(width: 224.0, height: 224.0)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
