//
//  BlenderbotTokenizer.swift
//  huggingface-ios-app
//
//  Created by Daniel Vela on 10/5/23.
//
// this file was created by shamelessly copying the GPT2Tokenizer.swift created by Julien Chaumond

import Foundation

class BlenderbotTokenizer {
    let bpeRanks: Dictionary<BytePair, Int>
    private let encoder: [String: Int]
    private let decoder: [Int: String]
    
    init(urlMerges: URL, urlVocab: URL) throws {
        let bpeMergesTxt = try String(contentsOf: urlMerges)
        let arr = bpeMergesTxt.split(separator: "\n").map { String($0) }
        var bpeRanks: Dictionary<BytePair, Int> = [:]
        for i in 1..<arr.count {
            let tuple = arr[i].split(separator: " ").map { String($0) }
            let bp = BytePair(tuple: tuple)
            bpeRanks[bp] = i - 1
        }
        self.bpeRanks = bpeRanks
        
        self.encoder = try {
            let json = try Data(contentsOf: urlVocab)
            let decoder = JSONDecoder()
            let vocab = try! decoder.decode([String: Int].self, from: json)
            return vocab
        }()
        self.decoder = Utils.invert(self.encoder)
    }
    
    static func from_pretrained(_ model_name: String, cache_dir: String? = nil) throws -> BlenderbotTokenizer {
        guard let urlMerges = Bundle.main.url(forResource: model_name + "/merges", withExtension: "txt"),
              let urlVocab = Bundle.main.url(forResource: model_name + "/vocab", withExtension: "json") else {
            throw TokenizerError.invalidModelName(model_name)
        }
        
        return try BlenderbotTokenizer(urlMerges: urlMerges, urlVocab: urlVocab)
    }
    
    func byteEncode(text: String) -> [String] {
        let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let tokens = text.ranges(of: RE).map { String(text[$0]) }
        return tokens.map { (token) -> String in
            return Array(token.utf8).map { byteEncoder[$0]! }.joined()
        }
    }
    
    private func getPairs(word: [String]) -> Set<BytePair> {
        var s = Set<BytePair>()
        for i in 0..<word.count-1 {
            let bp = BytePair(
                word[i],
                word[i+1]
            )
            s.insert(bp)
        }
        return s
    }
    
    func bpe(token: String) -> String {
        if token.count <= 1 {
            return token
        }
        
        var word = Array(token).map { String($0) }
        var pairs = Array(getPairs(word: word))
        
        while true {
            let bigrams = pairs.filter { (bp) -> Bool in bpeRanks[bp] != nil }
            if bigrams.count == 0 {
                break
            }
            let bigram = bigrams.min { (bp1, bp2) -> Bool in
                return bpeRanks[bp1]! < bpeRanks[bp2]!
            }!
            let first = bigram.a
            let second = bigram.b
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if let j = word[i..<word.count].firstIndex(of: first) {
                    newWord.append(contentsOf: word[i..<j])
                    i = j
                } else {
                    newWord.append(contentsOf: word[i..<word.count])
                    break
                }
                
                if word[i] == first && i < word.count - 1 && word[i+1] == second {
                    newWord.append(first+second)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
            if word.count == 1 {
                break
            } else {
                pairs = Array(getPairs(word: word))
            }
        }
        return word.joined(separator: " ")
    }
    
    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        for token in self.byteEncode(text: text) {
            let xx = self.bpe(token: token).split(separator: " ").map { String($0) }
            tokens.append(contentsOf: xx)
        }
        return tokens + ["</s>"]
    }
    
    /// Main entry point
    func encode(text: String) -> [Int] {
        return tokenize(text: text).map { encoder[$0]! }
    }
    
    /// Decode
    func decode(tokens: [Int]) -> String {
        let text = tokens.compactMap {
            if $0 != 2 { // end string token
                return decoder[$0]!
            }
            return nil
        }.joined(separator: "")
        let utfCodepoints = text.map { byteDecoder[String($0)]! }
        return String(decoding: utfCodepoints, as: UTF8.self)
    }
}
