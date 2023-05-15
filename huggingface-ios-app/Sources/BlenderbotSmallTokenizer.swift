//
//  BlenderbotSmallTokenizer.swift
//  huggingface-ios-app
//
//  Created by Daniel Vela on 11/5/23.
//

import Foundation

fileprivate struct BytePair: Hashable {
    let a: String
    let b: String
    init(_ a: String, _ b: String) {
        self.a = a
        self.b = b
    }
    init(tuple: [String]) {
        self.a = tuple[0]
        self.b = tuple[1]
    }
    
    static func == (lhs: BytePair, rhs: BytePair) -> Bool {
        return lhs.a == rhs.a && lhs.b == rhs.b
    }
    func hash(into hasher: inout Hasher) {
        hasher.combine(a)
        hasher.combine(b)
    }
}

fileprivate extension String {
    func ranges(of string: String, options: CompareOptions = .regularExpression) -> [Range<Index>] {
        var result: [Range<Index>] = []
        var start = startIndex
        while let range = range(of: string, options: options, range: start..<endIndex) {
            result.append(range)
            start = range.lowerBound < range.upperBound ? range.upperBound : index(range.lowerBound, offsetBy: 1, limitedBy: endIndex) ?? endIndex
        }
        return result
    }
}

class BlenderbotSmallTokenizer {
    fileprivate let bpeRanks: Dictionary<BytePair, Int>
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
    
    static func from_pretrained(_ model_name: String, cache_dir: String? = nil) throws -> BlenderbotSmallTokenizer {
        guard let urlMerges = Bundle.main.url(forResource: model_name + "/merges", withExtension: "txt"),
              let urlVocab = Bundle.main.url(forResource: model_name + "/vocab", withExtension: "json") else {
            throw TokenizerError.invalidModelName(model_name)
        }
        
        return try BlenderbotSmallTokenizer(urlMerges: urlMerges, urlVocab: urlVocab)
    }
    
    func byteEncode(text: String) -> [String] {
        let range = NSRange(location: 0, length: text.utf16.count)
        let pattern = #"\S+\n?"#
        let regex = try! NSRegularExpression(pattern: pattern)
        let tokens = regex.matches(in: text, range: range).map { String(text[Range($0.range, in: text)!]) }.map { $0.lowercased() }
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
        var token = token.replacingOccurrences(of: "([.,!?()])", with: " $1", options: .regularExpression)
        token = token.replacingOccurrences(of: "(')", with: " $1 ", options: .regularExpression)
        token = token.replacingOccurrences(of: "\\s{2,}", with: " ", options: .regularExpression)
        if token.contains("\n") {
            token = token.replacingOccurrences(of: "\n", with: " __newln__")
        }

        let tokens = token.split(separator: " ")
        var words = [String]()
        for token in tokens {
            guard !token.isEmpty else {
                continue
            }
            var token = token.lowercased()
            var word = Array(token).map { String($0) }
            word = Array(word[0..<word.count - 1]) + [word[word.count - 1] + "</w>"]
            var pairs = getPairs(word: word)

            if pairs.isEmpty {
                words.append(token)
                continue
            }

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
                var newWord = [String]()
                var i = 0

                while i < word.count {
                    if let j = word[i..<word.count].firstIndex(of: first) {
                        newWord.append(contentsOf: word[i..<j])
                        i = j
                    } else {
                        newWord.append(contentsOf: word[i..<word.count])
                        break
                    }

                    if word[i] == first && i < word.count - 1 && word[i + 1] == second {
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
                    pairs = getPairs(word: word)
                }
            }
            let word2 = word.map { String($0) }.joined(separator: "@@ ")
            words.append(String(word2.dropLast(4)))
        }
        return words.joined(separator: " ")
    }
    
    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        for token in self.byteEncode(text: text) {
            let xx = self.bpe(token: token).split(separator: " ").map { String($0) }
            tokens.append(contentsOf: xx)
        }
        return tokens
    }
    
    /// Main entry point
    func encode(text: String) -> [Int] {
        return tokenize(text: text).map { encoder[$0]! }
    }
    
    /// Decode
    func decode(tokens: [Int]) -> String {
        let text = tokens.map { decoder[$0]! }.joined(separator: "")
        let utfCodepoints = text.map { byteDecoder[String($0)]! }
        return String(decoding: utfCodepoints, as: UTF8.self)
    }
}
