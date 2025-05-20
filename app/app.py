import os
from flask import Flask, redirect, request, session, url_for, render_template, flash
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

# Spotify API scopes
SCOPE = "user-library-read user-top-read playlist-modify-public playlist-modify-private"

# Read Spotify API credentials from environment variables
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")

sp_oauth = SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope=SCOPE,
    cache_path=".spotifycache"  # this can remain but not mandatory if you use session
)

def get_spotify_client():
    token_info = session.get("token_info")
    if not token_info:
        return None

    # Refresh token if expired
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info["refresh_token"])
        session["token_info"] = token_info

    return Spotify(auth=token_info["access_token"])

def fetch_liked_songs(sp):
    results = []
    limit = 50
    offset = 0
    while True:
        liked = sp.current_user_saved_tracks(limit=limit, offset=offset)
        results.extend(liked['items'])
        if liked['next'] is None:
            break
        offset += limit
    return results

def fetch_top_songs(sp):
    results = []
    limit = 50
    offset = 0
    while True:
        top = sp.current_user_top_tracks(limit=limit, offset=offset, time_range='medium_term')
        results.extend(top['items'])
        if top['next'] is None:
            break
        offset += limit
    return results

def get_audio_features(sp, track_ids):
    features = []
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i+100]
        features.extend(sp.audio_features(batch))
    return features

def cluster_and_create_playlists(sp, tracks, user_id, cluster_count=2):
    if not tracks:
        return None

    track_ids = [t['track']['id'] if 'track' in t else t['id'] for t in tracks]
    features = get_audio_features(sp, track_ids)

    features = [f for f in features if f]

    X = np.array([[f['danceability'], f['energy'], f['valence'], f['tempo']] for f in features])

    kmeans = KMeans(n_clusters=cluster_count, random_state=42)
    labels = kmeans.fit_predict(X)

    clustered_tracks = {i: [] for i in range(cluster_count)}
    for idx, label in enumerate(labels):
        clustered_tracks[label].append(track_ids[idx])

    created_playlists = []
    for cluster_label, track_list in clustered_tracks.items():
        playlist_name = f"Cluster {cluster_label + 1} Playlist"
        playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=False)
        for i in range(0, len(track_list), 100):
            sp.playlist_add_items(playlist_id=playlist['id'], items=track_list[i:i+100])
        created_playlists.append(playlist_name)
    return created_playlists

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login")
def login():
    auth_url = sp_oauth.get_authorize_url()
    print(f"Auth URL: {auth_url}")
    return redirect(auth_url)

@app.route("/callback")
def callback():
    code = request.args.get('code')
    error = request.args.get('error')

    print(f"[DEBUG] Callback received. Code: {code}, Error: {error}")

    if error:
        flash(f"Spotify authorization failed: {error}", "danger")
        return redirect(url_for("index"))

    if code:
        token_info = sp_oauth.get_access_token(code)
        print(f"[DEBUG] Token info obtained: {token_info}")
        session["token_info"] = token_info
        return redirect(url_for("choose"))

    flash("Authorization code missing", "danger")
    return redirect(url_for("index"))

@app.route("/choose", methods=["GET", "POST"])
def choose():
    sp = get_spotify_client()
    if not sp:
        flash("Please login first.", "warning")
        return redirect(url_for("index"))

    if request.method == "POST":
        choice = request.form.get("song_choice")
        user_id = sp.current_user()["id"]

        tracks = []
        if choice == "liked" or choice == "both":
            tracks.extend(fetch_liked_songs(sp))
        if choice == "top" or choice == "both":
            tracks.extend(fetch_top_songs(sp))

        if not tracks:
            flash("No songs found in the selected category.", "warning")
            return redirect(url_for("choose"))

        playlists = cluster_and_create_playlists(sp, tracks, user_id)
        flash(f"Created playlists: {', '.join(playlists)}", "success")
        return redirect(url_for("index"))

    return render_template("choose.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
