(ns vr3d.core
  (:require [quil.core :as q]
            [quil.middleware :as m]
            [langohr.core :as rmq]
            [langohr.channel :as lch]
            [langohr.queue :as lq]
            [langohr.consumers :as lc]
            [langohr.basic :as lb]))

; Draws sphere at point [0 0 0] and 6 cubes around it.
; You can fly around this objects using navigation-3d.
; This draw function is fun-mode compatible (it takes state),
                                        ; though it's not used here, but we need fun-mode for navigation-3d.
(defn setup []
  (q/frame-rate 60)
  {:fishx 0
   :fishy 0})

(defn update-state [state]
  {:fishx (+ 1 (:fishx state))
   :fishy (+ 0 (:fishy state))})

(defn draw [state]
  (q/background 255)
  (q/lights)
  (q/fill 150 100 150)
  (q/with-translation [0 0 0]
    (q/sphere 75))
  (q/camera (:fishx state) (:fishy state) 0 0 0 0 0 0 1)
  )


(q/defsketch my-sketch
  :draw draw
  :size [500 500]
  :renderer :p3d
  :update update-state
  :setup setup
  ; Enable navigation-3d.
  ; Note: it should be used together with fun-mode.
  :middleware [m/fun-mode])
