"""
Netflix Benzeri İçerik Tabanlı (Content-Based) Film/Dizi Öneri Sistemi
======================================================================

Bu proje, TF-IDF vektörizasyonu ve Kosinüs Benzerliği kullanarak
verilen bir film/diziye benzer içerikleri öneren bir sistemdir.

Kullanılan Teknikler:
---------------------
- TF-IDF (Term Frequency - Inverse Document Frequency):
    Metin verisini sayısal vektörlere dönüştürür. Bir kelimenin bir belgede
    ne kadar önemli olduğunu ölçer. Sık geçen ama her yerde bulunan kelimeler
    (ör. "the", "and") düşük ağırlık alırken, belgeye özgü kelimeler yüksek
    ağırlık alır.

- Cosine Similarity (Kosinüs Benzerliği):
    İki vektör arasındaki açının kosinüsünü hesaplayarak benzerlik ölçer.
    0 = hiç benzemez, 1 = tamamen aynı. Metin karşılaştırmada çok
    etkilidir çünkü belge uzunluğundan bağımsız çalışır.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─── Örnek Veri Seti ───────────────────────────────────────────────
# Netflix Movies and TV Shows benzeri bir veri seti oluşturuyoruz.
# Gerçek projede CSV dosyasından yüklenebilir: pd.read_csv("netflix_titles.csv")

SAMPLE_DATA = [
    {"title": "Stranger Things", "genre": "Sci-Fi, Horror, Drama", "description": "When a young boy vanishes, a small town uncovers a mystery involving secret experiments, terrifying supernatural forces, and one strange little girl."},
    {"title": "The Witcher", "genre": "Fantasy, Action, Adventure", "description": "Geralt of Rivia, a mutated monster-hunter for hire, journeys toward his destiny in a turbulent world where people often prove more wicked than beasts."},
    {"title": "Dark", "genre": "Sci-Fi, Thriller, Mystery", "description": "A family saga with a supernatural twist, set in a German town where the disappearance of two young children exposes the relationships among four families."},
    {"title": "Breaking Bad", "genre": "Crime, Drama, Thriller", "description": "A high school chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing and selling methamphetamine to secure his family's future."},
    {"title": "Narcos", "genre": "Crime, Drama, Thriller", "description": "A chronicled look at the criminal exploits of Colombian drug lord Pablo Escobar, as well as the many other drug kingpins who plagued the country."},
    {"title": "Black Mirror", "genre": "Sci-Fi, Drama, Thriller", "description": "An anthology series exploring a twisted high-tech multiverse where humanity's greatest innovations and darkest instincts collide."},
    {"title": "The OA", "genre": "Sci-Fi, Fantasy, Drama", "description": "Having gone missing seven years ago, the previously blind Prairie Johnson returns home with her sight restored and tells stories of supernatural experiences."},
    {"title": "Ozark", "genre": "Crime, Drama, Thriller", "description": "A financial adviser drags his family from Chicago to the Missouri Ozarks, where he must launder money to appease a drug boss."},
    {"title": "Money Heist", "genre": "Action, Crime, Mystery", "description": "An unusual group of robbers attempt to carry out the most perfect robbery in Spanish history, stealing billions from the Royal Mint of Spain."},
    {"title": "The Haunting of Hill House", "genre": "Horror, Drama, Mystery", "description": "Flashing between past and present, a fractured family confronts haunting memories of their old home and the terrifying events that drove them from it."},
    {"title": "Mindhunter", "genre": "Crime, Drama, Thriller", "description": "Set in the late 1970s, two FBI agents are tasked with interviewing serial killers to solve open cases."},
    {"title": "Peaky Blinders", "genre": "Crime, Drama", "description": "A gangster family epic set in Birmingham England in 1919, centering on a gang who sew razor blades in the peaks of their caps."},
    {"title": "Altered Carbon", "genre": "Sci-Fi, Action, Drama", "description": "Set in a future where consciousness is digitized and stored, a prisoner returns to life in a new body and must solve a mind-bending murder."},
    {"title": "The Umbrella Academy", "genre": "Action, Adventure, Comedy", "description": "A family of former child heroes, now combative adults, reunite to solve the mystery of their father's death and prevent an apocalypse."},
    {"title": "Squid Game", "genre": "Action, Drama, Mystery", "description": "Hundreds of cash-strapped players accept a strange invitation to compete in children's games for a tempting prize, but the stakes are deadly."},
    {"title": "The Crown", "genre": "Biography, Drama, History", "description": "Follows the political rivalries and romance of Queen Elizabeth II's reign and the events that shaped the second half of the twentieth century."},
    {"title": "Lucifer", "genre": "Crime, Drama, Fantasy", "description": "Bored and unhappy as the Lord of Hell, Lucifer Morningstar abandons his throne and retires to Los Angeles, where he helps the LAPD punish criminals."},
    {"title": "Westworld", "genre": "Sci-Fi, Drama, Mystery", "description": "Set at the intersection of the near future and the reimagined past, it explores a world where every human appetite can be indulged in an advanced theme park."},
    {"title": "You", "genre": "Crime, Drama, Romance", "description": "A dangerously charming obsessive young man goes to extreme measures to insert himself into the lives of those he is transfixed by."},
    {"title": "Alice in Borderland", "genre": "Action, Drama, Sci-Fi", "description": "A gamer and his friends find themselves in an abandoned Tokyo where they must compete in dangerous games to survive."},
]


def load_and_preprocess(data: list[dict]) -> pd.DataFrame:
    """Veri setini yükler ve ön işleme yapar."""
    df = pd.DataFrame(data)

    # Eksik description alanlarını temizle
    df["description"] = df["description"].fillna("")
    df["genre"] = df["genre"].fillna("")

    # Genre ve description'ı birleştirerek zengin bir metin özelliği oluştur
    df["combined_features"] = df["genre"] + " " + df["description"]
    df["combined_features"] = df["combined_features"].str.lower().str.strip()

    return df


def build_similarity_matrix(df: pd.DataFrame):
    """
    TF-IDF vektörizasyonu uygular ve kosinüs benzerlik matrisini hesaplar.

    TF-IDF: Her kelimenin önemini belge bazında ölçer.
            stop_words='english' → yaygın İngilizce kelimeleri filtreler.

    Cosine Similarity: Tüm içerik çiftleri arasındaki benzerliği hesaplar.
                       Sonuç NxN boyutunda bir matristir (N = içerik sayısı).
    """
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])

    # Kosinüs benzerlik matrisi: her içeriğin diğer tüm içeriklerle benzerliği
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return similarity_matrix


def recommend(title: str, df: pd.DataFrame, similarity_matrix, top_n: int = 5) -> list[str]:
    """
    Verilen bir film/dizi başlığına en benzer top_n içeriği önerir.

    Nasıl çalışır:
    1. Girilen başlığın veri setindeki indeksini bulur
    2. Bu başlığın tüm diğer içeriklerle benzerlik skorlarını alır
    3. Skorları büyükten küçüğe sıralar
    4. En benzer top_n içeriği döndürür (kendisi hariç)
    """
    # Başlık kontrolü (büyük/küçük harf duyarsız)
    title_lower = title.lower()
    matches = df[df["title"].str.lower() == title_lower]

    if matches.empty:
        raise ValueError(
            f"'{title}' bulunamadı. Mevcut başlıklar:\n"
            + "\n".join(f"  - {t}" for t in sorted(df["title"].tolist()))
        )

    idx = matches.index[0]

    # Benzerlik skorlarını al ve sırala
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # İlk eleman kendisi olduğu için atlıyoruz
    top_matches = sim_scores[1 : top_n + 1]

    results = []
    for i, score in top_matches:
        results.append(f"{df.iloc[i]['title']}  (benzerlik: {score:.2f})")

    return results


def main():
    """Ana fonksiyon: veriyi yükler, model kurar ve terminal arayüzü sağlar."""
    print("=" * 60)
    print("  🎬 Netflix Benzeri Film/Dizi Öneri Sistemi")
    print("=" * 60)

    df = load_and_preprocess(SAMPLE_DATA)
    similarity_matrix = build_similarity_matrix(df)

    print(f"\n✅ {len(df)} içerik yüklendi ve model hazır.\n")
    print("Mevcut içerikler:")
    for t in sorted(df["title"].tolist()):
        print(f"  • {t}")

    # Terminal arayüzü
    while True:
        print("\n" + "-" * 40)
        user_input = input("Film/dizi adı girin (çıkmak için 'q'): ").strip()

        if user_input.lower() in ("q", "quit", "exit"):
            print("Hoşça kalın! 👋")
            break

        if not user_input:
            continue

        try:
            recommendations = recommend(user_input, df, similarity_matrix)
            print(f"\n🎯 '{user_input}' için öneriler:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        except ValueError as e:
            print(f"\n⚠️  {e}")


if __name__ == "__main__":
    main()
