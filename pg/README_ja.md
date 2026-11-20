# kci_vec_pg — PostgreSQL の中から KCI-Vec を使う (試作)

KCI-Vec を SQL の関数として呼べます (`CREATE EXTENSION kci_vec_pg`)。サーバー・権限・バックアップ・複製は PostgreSQL のものをそのまま使います。
**試作です**: 関数と目録の表は、まだ変わることがあります。

**pgvector と同じ書き方ではありません。** `CREATE INDEX` も `<=>` もなく、関数で呼びます。
**自分で管理する PostgreSQL 向けです。** 拡張を自由に入れられない管理型のサービス (RDS、Cloud SQL、Supabase など) では使えません。

## 組み立てと導入

`pg_config` (`postgresql-server-dev-NN`)、gcc、このリポジトリのファイルが要ります。

```sh
cd pg && make && sudo make install            # または: make PG_CONFIG=/path/to/pg_config install
```

任意で `postgresql.conf` に (そのあと再起動):

```
shared_preload_libraries = 'kci_vec_pg'      # 更新の追跡が軽くなる。裏方 (背景ワーカー) にも要る
kci_vec_pg.databases = 'mydb'                # 裏方: プライマリでは自動で作り直し、スタンバイでは無いファイルを作る
kci_vec_pg.refresh_pending = 10000           # 溜まった更新がこの件数で作り直す
```

PostgreSQL 15.19 / 16.15 / 17.11 (Linux x86-64) で試験しています。

## 使い方

```sql
CREATE EXTENSION vector;                     -- 任意: pgvector の vector 型をそのまま渡せる
CREATE EXTENSION kci_vec_pg;

-- 表 docs(id bigint, embedding vector(384) または real[]) から、索引 'docs' を作る
SELECT kci_vec_build('docs', 'docs', 'id', 'embedding');
SELECT kci_vec_track('docs');                -- 以後の INSERT / UPDATE / DELETE / TRUNCATE を追う

-- 近い 10 件 (近い順)
SELECT id, score FROM kci_vec_search('docs', '[0.1, ...]'::vector, 10);
-- この id の中からだけ (WHERE 句を SQL で評価して渡す)
SELECT id, score FROM kci_vec_search('docs', '[0.1, ...]'::vector, 10, 64, 200,
                                     ARRAY(SELECT id FROM docs WHERE tenant = 7));

SELECT * FROM kci_vec_status();              -- 件数、溜まった更新、このサーバーに索引のファイルがあるか
SELECT kci_vec_refresh('docs');              -- 溜まった更新を索引に入れる
```

| 関数 | すること |
|---|---|
| `kci_vec_build(name, rel, id_col, vec_col, nlist=0, threads=0, train_iters=0, train_sample=0, metric='cosine')` | 表から索引を作る。`metric` は `'cosine'`・`'l2'` (ユークリッド)・`'ip'` (内積)。入れた行数を返す |
| `kci_vec_search(name, q, k=10, nprobe=32, rerank=100)` | `(id bigint, score real)` を近い順に返す。`score` は cosine・内積ならそのまま (大きいほど近い)、`'l2'` は距離の 2 乗 (小さいほど近い)。`nprobe`・`rerank` を大きくすると正確になり、遅くなる |
| `kci_vec_search(name, q, k, nprobe, rerank, allowed bigint[])` | 同じで、`allowed` の id の中からだけ |
| `kci_vec_track(name)` / `kci_vec_untrack(name)` | 表の更新を追い始める / やめる。追っている間、検索は更新をすぐ見る |
| `kci_vec_refresh(name, threads=0, full=false)` | 溜まった更新を索引に入れる |
| `kci_vec_status()` | 索引ごとに 1 行 |
| `kci_vec_materialize(name, threads=0)` | このサーバーの索引のファイルを、手元の表から作る (スタンバイ、ダンプから戻したあと) |
| `kci_vec_image_here(name)` | いまの索引のファイルがこのサーバーにあるか |
| `kci_vec_set_probe_gap(name, gap, min_probe=8)` | 検索の設定 (この接続だけ) |
| `kci_vec_count(name)` / `kci_vec_drop(name)` | 索引の行数 / 索引を消す |

id の列は `bigint` に、ベクトルの列は `real[]` に変換できること。id かベクトルが NULL の行は飛ばします。

## 知っておくこと

- **追っている表の更新は、次の検索に反映されます** (自分の開いたトランザクションの中でも)。更新は `kci_vec_refresh` (または裏方) が
  索引に入れるまで普通の表に溜まり、溜まるほど検索は遅くなります。
- **スタンバイと復元**: 索引のファイルは複製されず、`pg_dump` にも入りません。スタンバイや復元したサーバーは、古いファイル・無いファイルでは
  検索を断り、`kci_vec_materialize` を名指しします (手元の表から作ります。スタンバイでは裏方が自分で行います)。
  `kci_vec_build` と `kci_vec_refresh` はプライマリだけで動きます。
- **索引のファイルは暗号化していません** (`$PGDATA/kci_vec/`)。守られ方は PostgreSQL の表のファイルと同じです。ディスクの暗号化と一緒に使ってください。
  `DROP EXTENSION` はファイルを消しません。先に `kci_vec_drop` を呼んでください。
- **`'l2'` と `'ip'`**: 作るときに表を 2 回読みます。あとから入った行が、作ったときの一番長いベクトルより長いと、次の作り直しは全体になります。
  長さのそろったデータ向きです (飛び抜けて長いベクトルが少しあるだけで、同じ設定での正確さが下がります)。
- **絞り込み**は、許す id の配列で渡します。小さな集合 (全体の 1 % 前後かそれ以下) では `nprobe` を大きくしてください。数千件以下なら、
  SQL の `ORDER BY` で正確に並べるほうが向いています。
- **拡張の更新**: 新しいファイルを入れ、サーバーを再起動し、`ALTER EXTENSION kci_vec_pg UPDATE`。作ってある索引はそのまま残ります。
- 無いもの: `CREATE INDEX` と演算子 (planner が自分で選ぶことはありません)、索引のファイルの複製、その暗号化。

## pgvector の HNSW との比べ

同じ PostgreSQL 16.15 の中で、同じ表を、同じ SQL の往復 (準備済みの文、1 問ずつ、1 接続) で測りました。glove-100-angular
(1,183,514 行 x 100 次元、cosine)、10,000 問、k = 10、recall@10 はデータ付属の正確な答えと比べます。pgvector 0.8.1 の HNSW (m 16、
ef_construction 200)。Ryzen 7 8845HS、WSL2、`shared_buffers` 6 GB、2026-11-20、交互に 3 巡の最小〜最大。

| | recall@10 | 問/秒 |
|---|---|---|
| kci_vec_pg、nprobe 64 / rerank 200 | 0.9005 | 3,261〜3,906 |
| pgvector HNSW、ef_search 240 | 0.8947 | 199〜210 |
| kci_vec_pg、nprobe 200 / rerank 600 | 0.9632 | 1,776〜2,206 |
| pgvector HNSW、ef_search 1000 (上限) | 0.9612 | 58〜59 |
| SQL の往復だけ | | 9,549〜11,622 |

- 開発者自身の計測です。1 つのデータ、1 台の機械、1 接続です。
- **この版で測っていないもの**: `'l2'`・`'ip'` の索引、同時の接続、絞り込み、ほかのデータ、PostgreSQL 15 / 17。
- pgvector の索引は 1 度作ったもの (2026-11-08、7 ワーカーで 373 秒) を使い、今回は作り直していません。kci_vec_pg の索引は毎巡作り直しました (8 スレッドで 12 秒)。
- pgvector は、この拡張が使えない所で使えます: 管理型のサービス、`CREATE INDEX`、planner が分かる演算子、WAL による索引の複製。
  行の 1 割ほどかそれ以上を残す絞り込みでは、以前の計測 (2026-11-09) で pgvector 自身の `WHERE` のほうが速い結果でした
  (この拡張は、先に許す id の配列を受け取る必要があるため)。
- 台本は `pg/bench_pg0.py` です。
