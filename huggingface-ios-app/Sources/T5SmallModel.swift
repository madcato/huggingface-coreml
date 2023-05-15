//
//  T5SmallModel.swift
//  huggingface-ios-app
//
//  Created by Daniel Vela on 13/5/23.
//

import Foundation
import CoreML

class T5SmallModel {
    let encoder: encoder_safetybot_t5_base
    let decoder: decoder_safetybot_t5_base
//    let tokenizer: BlenderbotSmallTokenizer
    let seqLen = 128
    
    init() throws {
//        let modelName = "facebook/blenderbot-small-90M"
//        self.tokenizer = try BlenderbotSmallTokenizer.from_pretrained(modelName)
        self.encoder = try encoder_safetybot_t5_base()
        self.decoder = try decoder_safetybot_t5_base()
    }
    
    func predict(_ string: String) throws -> String {
//        let tokens = tokenizer.encode(text: string)
//        let tokens = [1] + tokens_0 + [2]
//        let tokens = [8774,  296,  149,   33,   25, 58,   1, 0 , 0, 0, 0, 0, 0,0, 0,0 ,0, 0,0 ,0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0]
        let tokens = [ 499,  564,   19, 4173,    5,  363,   31,    7,   39,  564,   58,    10 ]

        let maxTokens = (tokens.count > seqLen)
            ? Array(tokens[..<seqLen])
            : tokens

        let inputArray = MLMultiArray.from(maxTokens, dims: 2)
        let maskArray = MLMultiArray.from(
            Array(repeating: 1, count: maxTokens.count)
            , dims: 2
        )
        let inputEncoder = encoder_safetybot_t5_baseInput(input_ids: inputArray, attention_mask: maskArray)
        
        let outputEncoder = try encoder.prediction(input: inputEncoder)
        
        let outputDecoder = try decoder.prediction(decoder_input_ids: inputArray, decoder_attention_mask: maskArray, encoder_last_hidden_state: outputEncoder.last_hidden_state, encoder_attention_mask: maskArray)
        
        
        for i in 0..<outputDecoder.token_scores.shape[0].intValue {
            for j in 0..<outputDecoder.token_scores.shape[1].intValue {
                let outputLogits = MLMultiArray.slice(
                    outputDecoder.token_scores,
                    indexing: [.select(i), .select(j), .slice]
                )

                let nextToken = Math.argmax(outputLogits)
                let tok = nextToken.0
                
                print(tok)
                print(", ")
            }
        }
        
        
//        let outputLogits = MLMultiArray.slice(
//            outputDecoder.token_scores,
//            indexing: [.select(0), .select(tokens.count - 1), .slice]
//        )
//
//        let nextToken = Math.argmax(outputLogits)
//        let tok = nextToken.0
//
//        print(tok)
        
        return ""
    }
}
