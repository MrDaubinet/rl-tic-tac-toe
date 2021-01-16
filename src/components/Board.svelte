<script>
  // Component
  import Marker from "../components/Marker.svelte"
  // Data prop
  export let states = [0, 0, 0, 0, 0, 0, 0, 0, 0]
  // Function prop
  export let selectMarker
</script>

<div class="grid grid-cols-3 gap-0">
  {#each states as state, index}
    <div class="col-span-1">
      <Marker 
        {state}
        {index}
        {selectMarker}
      />
    </div>
  {/each}
</div>

